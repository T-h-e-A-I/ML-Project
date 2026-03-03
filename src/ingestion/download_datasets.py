"""Download and preprocess ScienceQA and AI2D datasets from HuggingFace."""

import json
import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from configs.default import DATA_RAW, DATA_PROCESSED, DATA_EVAL

PHYSICS_KEYWORDS = {
    "force", "motion", "energy", "momentum", "gravity", "acceleration",
    "velocity", "friction", "circuit", "voltage", "current", "resistance",
    "wave", "frequency", "optics", "lens", "mirror", "reflection",
    "refraction", "thermodynamics", "heat", "temperature", "pressure",
    "magnetic", "electric", "field", "potential", "kinetic", "work",
    "power", "torque", "rotation", "oscillation", "pendulum", "spring",
    "fluid", "density", "buoyancy", "projectile", "incline", "pulley",
}


def download_scienceqa():
    """Download ScienceQA and filter for physics-related multimodal questions."""
    print("Downloading ScienceQA dataset...")
    ds = load_dataset("derek-thomas/ScienceQA", trust_remote_code=True)

    output_dir = DATA_RAW / "scienceqa"
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    physics_samples = []

    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue
        for idx, sample in enumerate(tqdm(ds[split], desc=f"ScienceQA {split}")):
            if sample.get("image") is None:
                continue

            hint_text = (sample.get("hint", "") or "").lower()
            question_text = (sample.get("question", "") or "").lower()
            combined = hint_text + " " + question_text

            if not any(kw in combined for kw in PHYSICS_KEYWORDS):
                continue

            img_filename = f"{split}_{idx}.png"
            img_path = img_dir / img_filename
            if isinstance(sample["image"], Image.Image):
                sample["image"].save(str(img_path))

            choices = sample.get("choices", [])
            answer_idx = sample.get("answer", 0)

            physics_samples.append({
                "id": f"scienceqa_{split}_{idx}",
                "split": split,
                "image_path": str(img_path),
                "question": sample.get("question", ""),
                "choices": choices,
                "answer_idx": answer_idx,
                "answer": choices[answer_idx] if answer_idx < len(choices) else "",
                "hint": sample.get("hint", ""),
                "solution": sample.get("solution", ""),
                "topic": "physics",
                "source": "scienceqa",
            })

    out_file = output_dir / "physics_samples.json"
    with open(out_file, "w") as f:
        json.dump(physics_samples, f, indent=2)
    print(f"Saved {len(physics_samples)} physics samples to {out_file}")
    return physics_samples


def download_ai2d():
    """Download AI2D dataset of annotated science diagrams."""
    print("Downloading AI2D dataset...")
    ds = load_dataset("lmms-lab/ai2d", trust_remote_code=True)

    output_dir = DATA_RAW / "ai2d"
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    samples = []

    for split in ["test"]:
        if split not in ds:
            continue
        for idx, sample in enumerate(tqdm(ds[split], desc=f"AI2D {split}")):
            if sample.get("image") is None:
                continue

            img_filename = f"{split}_{idx}.png"
            img_path = img_dir / img_filename
            if isinstance(sample["image"], Image.Image):
                sample["image"].save(str(img_path))

            question = sample.get("question", "")
            options_raw = sample.get("options", [])
            answer = sample.get("answer", "")

            samples.append({
                "id": f"ai2d_{split}_{idx}",
                "split": split,
                "image_path": str(img_path),
                "question": question,
                "choices": options_raw if isinstance(options_raw, list) else [],
                "answer": answer,
                "topic": "science_diagram",
                "source": "ai2d",
            })

    out_file = output_dir / "ai2d_samples.json"
    with open(out_file, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved {len(samples)} AI2D samples to {out_file}")
    return samples


def build_eval_set():
    """Combine physics samples from both datasets into a unified eval set."""
    eval_dir = DATA_EVAL
    eval_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []

    scienceqa_path = DATA_RAW / "scienceqa" / "physics_samples.json"
    if scienceqa_path.exists():
        with open(scienceqa_path) as f:
            scienceqa = json.load(f)
        for s in scienceqa:
            all_samples.append({
                "id": s["id"],
                "image_path": s["image_path"],
                "question": s["question"],
                "answer": s["answer"],
                "reasoning": s.get("solution", ""),
                "topic": s.get("topic", "physics"),
                "difficulty": "medium",
                "source": "scienceqa",
            })

    ai2d_path = DATA_RAW / "ai2d" / "ai2d_samples.json"
    if ai2d_path.exists():
        with open(ai2d_path) as f:
            ai2d = json.load(f)
        for s in ai2d:
            all_samples.append({
                "id": s["id"],
                "image_path": s["image_path"],
                "question": s["question"],
                "answer": s["answer"],
                "reasoning": "",
                "topic": s.get("topic", "science_diagram"),
                "difficulty": "medium",
                "source": "ai2d",
            })

    out_file = eval_dir / "eval_dataset.json"
    with open(out_file, "w") as f:
        json.dump(all_samples, f, indent=2)
    print(f"Built unified eval set with {len(all_samples)} samples at {out_file}")
    return all_samples


if __name__ == "__main__":
    download_scienceqa()
    download_ai2d()
    build_eval_set()
