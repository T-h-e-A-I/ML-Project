"""Generate synthetic physics QA pairs using GPT-4o from diagrams."""

import base64
import json
import os
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from configs.default import DATA_EVAL, DATA_RAW, path_for_storage, resolve_data_path

SYSTEM_PROMPT = """You are an expert physics teacher creating exam questions from diagrams.
Given a physics diagram, generate a question-answer pair that requires visual reasoning.
The question should test understanding of the physical concepts shown in the diagram.

Return your response as JSON with these fields:
- question: The physics question about the diagram
- answer: The correct answer with brief explanation
- reasoning: Step-by-step reasoning to arrive at the answer
- topic: Physics topic (e.g., mechanics, circuits, optics, thermodynamics)
- difficulty: easy, medium, or hard
"""


def encode_image(image_path: str) -> str:
    """Encode image to base64 for the OpenAI API."""
    p = resolve_data_path(image_path)
    if p is None or not p.is_file():
        raise FileNotFoundError(image_path)
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_qa_from_image(
    client: OpenAI,
    image_path: str,
    model: str = "gpt-4o",
) -> dict | None:
    """Generate a QA pair from a single physics diagram using GPT-4o."""
    try:
        b64_image = encode_image(image_path)
        ext = Path(str(resolve_data_path(image_path) or image_path)).suffix.lstrip(".")
        mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Generate a physics question-answer pair for this diagram.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            max_tokens=1024,
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def generate_synthetic_dataset(
    image_dir: str | Path,
    output_path: str | Path | None = None,
    max_samples: int = 500,
):
    """Generate synthetic QA dataset from a directory of physics diagrams."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set. Skipping synthetic data generation.")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return []

    client = OpenAI(api_key=api_key)
    image_dir = Path(image_dir)

    if output_path is None:
        output_path = DATA_EVAL / "synthetic_physics_qa.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )[:max_samples]

    if not image_files:
        print(f"No images found in {image_dir}")
        return []

    print(f"Generating QA pairs for {len(image_files)} images...")
    samples = []

    for img_path in tqdm(image_files, desc="Generating QA"):
        qa = generate_qa_from_image(client, str(img_path))
        if qa:
            samples.append({
                "id": f"synthetic_{img_path.stem}",
                "image_path": path_for_storage(img_path),
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "reasoning": qa.get("reasoning", ""),
                "topic": qa.get("topic", "physics"),
                "difficulty": qa.get("difficulty", "medium"),
                "source": "synthetic_gpt4o",
            })

    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved {len(samples)} synthetic QA pairs to {output_path}")
    return samples


if __name__ == "__main__":
    img_dir = DATA_RAW / "physics_diagrams"
    if not img_dir.exists():
        img_dir = DATA_RAW / "scienceqa" / "images"
    generate_synthetic_dataset(img_dir)
