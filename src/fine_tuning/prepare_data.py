"""Prepare fine-tuning data for LLaVA QLoRA training."""

import json
import random
from pathlib import Path

from configs.default import DATA_RAW, DATA_EVAL, DATA_PROCESSED


def load_scienceqa_physics() -> list[dict]:
    """Load physics-filtered ScienceQA samples."""
    path = DATA_RAW / "scienceqa" / "physics_samples.json"
    if not path.exists():
        print(f"ScienceQA data not found at {path}")
        return []
    with open(path) as f:
        return json.load(f)


def load_synthetic_qa() -> list[dict]:
    """Load synthetic GPT-4o generated QA pairs."""
    path = DATA_EVAL / "synthetic_physics_qa.json"
    if not path.exists():
        print(f"Synthetic QA data not found at {path}")
        return []
    with open(path) as f:
        return json.load(f)


def format_for_llava(samples: list[dict]) -> list[dict]:
    """
    Convert samples to LLaVA conversational fine-tuning format.

    Each sample becomes:
    {
        "id": str,
        "image": str (path),
        "conversations": [
            {"from": "human", "value": "<image>\nQuestion text"},
            {"from": "gpt", "value": "Answer with reasoning"},
        ]
    }
    """
    formatted = []
    for s in samples:
        image_path = s.get("image_path", "")
        if not image_path or not Path(image_path).exists():
            continue

        question = s.get("question", "")
        answer = s.get("answer", "")
        reasoning = s.get("reasoning", "") or s.get("solution", "")

        if reasoning:
            full_answer = f"{reasoning}\n\nTherefore, the answer is: {answer}"
        else:
            full_answer = answer

        formatted.append({
            "id": s.get("id", f"sample_{len(formatted)}"),
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\nAnswer this physics question with step-by-step reasoning.\n\n{question}",
                },
                {
                    "from": "gpt",
                    "value": full_answer,
                },
            ],
        })

    return formatted


def create_splits(
    data: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """Split data into train/val/test sets."""
    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return {
        "train": data[:train_end],
        "val": data[train_end:val_end],
        "test": data[val_end:],
    }


def prepare_finetune_data(output_dir: Path | None = None):
    """Full pipeline: load, format, split, save."""
    if output_dir is None:
        output_dir = DATA_PROCESSED / "finetune"
    output_dir.mkdir(parents=True, exist_ok=True)

    scienceqa = load_scienceqa_physics()
    synthetic = load_synthetic_qa()

    all_samples = scienceqa + synthetic
    print(f"Total samples: {len(all_samples)} "
          f"(ScienceQA: {len(scienceqa)}, Synthetic: {len(synthetic)})")

    formatted = format_for_llava(all_samples)
    print(f"Formatted for LLaVA: {len(formatted)} samples (with valid images)")

    splits = create_splits(formatted)

    for split_name, split_data in splits.items():
        out_path = output_dir / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"  {split_name}: {len(split_data)} samples -> {out_path}")

    return splits


if __name__ == "__main__":
    prepare_finetune_data()
