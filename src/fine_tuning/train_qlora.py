"""QLoRA fine-tuning script for LLaVA-1.5-7B on physics diagrams.

Designed to run on Kaggle T4 GPUs (16GB VRAM) with 4-bit quantization.
"""

import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
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
):
    """Run QLoRA fine-tuning."""
    print("Loading training data...")
    train_data, val_data = load_training_data(data_dir)
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

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
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
    )

    data_collator = LlavaDataCollator(processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {output_dir}/final_adapter")
    model.save_pretrained(f"{output_dir}/final_adapter")
    processor.save_pretrained(f"{output_dir}/final_adapter")

    print("Training complete!")
    return model, processor


if __name__ == "__main__":
    train()
