"""QLoRA fine-tuning script for LLaVA-1.5-7B on physics diagrams.

Designed to run on Kaggle T4 GPUs (16GB VRAM) with 4-bit quantization.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from datasets import Dataset
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm.auto import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.default import (
    BATCH_SIZE,
    DATA_PROCESSED,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    NUM_EPOCHS,
    VLM_MODEL,
    resolve_data_path,
)
from src.fine_tuning.train_deps import ensure_qlora_train_deps


def estimate_qlora_training_steps(
    num_train_samples: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: int,
    world_size: int = 1,
) -> int:
    """Match Trainer's optimizer-step count (epoch-based runs, drop_last=False)."""
    if num_train_samples <= 0:
        return 0
    len_dataloader = math.ceil(num_train_samples / (per_device_batch_size * world_size))
    len_dataloader = max(len_dataloader, 1)
    num_update_steps_per_epoch = max(
        math.ceil(len_dataloader / gradient_accumulation_steps),
        1,
    )
    return num_update_steps_per_epoch * int(num_train_epochs)


class LoRATrainingProgressCallback(TrainerCallback):
    """Single tqdm bar for global training steps (clear in Kaggle / notebooks)."""

    def __init__(self, total_steps: int):
        self.total_steps = max(1, total_steps)
        self._pbar: tqdm | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        # bar_format forces a visible percentage (some Kaggle log views truncate the bar).
        self._pbar = tqdm(
            total=self.total_steps,
            desc="QLoRA train",
            unit="step",
            mininterval=2.0,
            dynamic_ncols=True,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} steps "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            ),
        )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._pbar is not None:
            self._pbar.n = min(int(state.global_step), self.total_steps)
            self._pbar.refresh()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self._pbar is not None:
            self._pbar.n = self.total_steps
            self._pbar.refresh()
            self._pbar.close()
            self._pbar = None
        return control


def load_training_data(data_dir: Path | None = None) -> tuple[list, list]:
    """Load train and val splits."""
    if data_dir is None:
        data_dir = DATA_PROCESSED / "finetune"

    train_path = data_dir / "train.json"
    val_path = data_dir / "val.json"

    with open(train_path) as f:
        train_data = json.load(f)
    with open(val_path) as f:
        val_data = json.load(f)

    return train_data, val_data


class LlavaDataCollator:
    """Custom data collator for LLaVA fine-tuning with images."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples: list[dict]) -> dict:
        texts = []
        images = []

        for ex in examples:
            convs = ex["conversations"]
            human_msg = convs[0]["value"]
            gpt_msg = convs[1]["value"]

            text = f"USER: {human_msg}\nASSISTANT: {gpt_msg}"
            texts.append(text)

            img_path = ex.get("image", "")
            resolved = resolve_data_path(img_path) if img_path else None
            if resolved and resolved.is_file():
                img = Image.open(resolved).convert("RGB")
                images.append(img)
            else:
                images.append(Image.new("RGB", (224, 224)))

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch


def train(
    output_dir: str = "outputs/llava_physics_qlora",
    data_dir: Path | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    num_train_epochs: int | None = None,
    max_steps: int | None = None,
):
    """Run QLoRA fine-tuning.

    Args:
        max_train_samples: Use only the first N train examples (faster dry runs).
        max_val_samples: Use only the first N val examples (faster eval).
        num_train_epochs: Override configs/default NUM_EPOCHS for this run.
        max_steps: If set, stop after this many optimizer steps (overrides epoch count).
    """
    ensure_qlora_train_deps()

    print("Loading training data...")
    train_data, val_data = load_training_data(data_dir)
    if max_train_samples is not None:
        train_data = train_data[: max(0, max_train_samples)]
    if max_val_samples is not None:
        val_data = val_data[: max(0, max_val_samples)]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    print("Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(VLM_MODEL)
    model = LlavaForConditionalGeneration.from_pretrained(
        VLM_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    epochs = int(num_train_epochs) if num_train_epochs is not None else NUM_EPOCHS
    cap_steps = int(max_steps) if max_steps is not None else -1

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=cap_steps,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.03,
        weight_decay=0.01,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        disable_tqdm=True,
    )

    _ws = max(1, int(getattr(training_args, "world_size", 1)))
    total_train_steps = estimate_qlora_training_steps(
        len(train_data),
        BATCH_SIZE,
        GRADIENT_ACCUMULATION_STEPS,
        epochs,
        world_size=_ws,
    )
    if cap_steps > 0:
        total_train_steps = min(total_train_steps, cap_steps)
    progress_cb = LoRATrainingProgressCallback(total_train_steps)
    print(
        f"Planned ~{total_train_steps} optimizer steps "
        f"({len(train_data)} samples, batch={BATCH_SIZE}, accum={GRADIENT_ACCUMULATION_STEPS}, epochs={epochs})."
    )

    data_collator = LlavaDataCollator(processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[progress_cb],
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {output_dir}/final_adapter")
    model.save_pretrained(f"{output_dir}/final_adapter")
    processor.save_pretrained(f"{output_dir}/final_adapter")

    print("Training complete!")
    return model, processor


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="QLoRA fine-tune LLaVA on physics diagrams.")
    p.add_argument(
        "--output-dir",
        default="outputs/llava_physics_qlora",
        help="Directory for checkpoints and final adapter.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Folder with train.json and val.json (default: processed/finetune).",
    )
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        metavar="N",
        help="Train on only the first N examples (time-budget runs).",
    )
    p.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate on only the first N validation examples.",
    )
    p.add_argument(
        "--num-train-epochs",
        type=int,
        default=None,
        metavar="E",
        help="Override NUM_EPOCHS from configs for this run (e.g. 1 for a quick pass).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        metavar="S",
        help="Stop after S optimizer steps (overrides full epoch schedule; caps wall time).",
    )
    args = p.parse_args()
    train(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
    )
