"""Fast predictions-only runner (no metrics).

Example:
  python -m src.evaluation.generate_predictions --config B1 \
    --max-samples 200 --max-new-tokens 128 \
    --output-preds data/preds/preds.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.default import DATA_EVAL, FUSION_ALPHA, TOP_K, resolve_data_path

CONFIGS = {
    "B1": {"name": "Text-only RAG + Mistral-7B", "retrieval": "text", "generator": "text_llm", "fine_tuned": False},
    "B2": {"name": "Text RAG + LLaVA-7B (no fine-tune)", "retrieval": "text", "generator": "vlm", "fine_tuned": False},
    "M1": {"name": "Multimodal RAG + LLaVA-7B", "retrieval": "multimodal", "generator": "vlm", "fine_tuned": False},
    "M2": {"name": "Multimodal RAG + LLaVA-7B (fine-tuned)", "retrieval": "multimodal", "generator": "vlm", "fine_tuned": True},
    "M3": {"name": "No retrieval + LLaVA-7B (fine-tuned)", "retrieval": "none", "generator": "vlm", "fine_tuned": True},
}


def load_eval_dataset(eval_path: Path | None = None) -> list[dict]:
    if eval_path is None:
        eval_path = DATA_EVAL / "eval_dataset.json"
    with open(eval_path, encoding="utf-8") as f:
        return json.load(f)


def _build_runner(config_key: str, adapter_path: str | None, b1_backend: str):
    cfg = CONFIGS[config_key]
    retriever = None
    if cfg["retrieval"] != "none":
        from src.retrieval.multimodal_retriever import MultimodalRetriever

        retriever = MultimodalRetriever()

    if cfg["generator"] == "text_llm":
        if b1_backend == "api":
            from src.generation.text_generator import APITextGenerator

            generator = APITextGenerator()
        else:
            from src.generation.text_generator import TextGenerator

            generator = TextGenerator()
    else:
        from src.generation.vlm_generator import VLMGenerator

        generator = VLMGenerator(adapter_path=adapter_path if cfg["fine_tuned"] else None)
    return cfg, retriever, generator


def run_predictions(
    config_key: str,
    eval_data: list[dict],
    output_preds: Path,
    adapter_path: str | None = None,
    top_k: int = TOP_K,
    alpha: float = FUSION_ALPHA,
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    use_cot: bool = False,
    b1_backend: str = "local",
) -> dict:
    cfg, retriever, generator = _build_runner(config_key, adapter_path, b1_backend)
    print(f"Running predictions-only: {config_key} - {cfg['name']}")
    print(f"Samples: {len(eval_data)} | max_new_tokens={max_new_tokens}, temp={temperature}")

    rows = []
    t0 = time.time()
    for i, sample in enumerate(
        tqdm(eval_data, desc=f"{config_key} preds", unit="ex", mininterval=2.0)
    ):
        question = sample["question"]
        reference = sample["answer"]
        raw_image = sample.get("image_path")
        query_img = resolve_data_path(raw_image) if raw_image else None

        context_chunks = []
        image_paths = []
        if retriever and cfg["retrieval"] == "text":
            context_chunks = retriever.retrieve_text_only(question, top_k=top_k)
        elif retriever and cfg["retrieval"] == "multimodal":
            ret = retriever.retrieve_multimodal(
                question,
                query_image_path=str(query_img) if query_img else None,
                top_k=top_k,
                alpha=alpha,
            )
            context_chunks = ret["text_results"]
            for r in ret["image_results"]:
                ip = r.get("image_path")
                rp = resolve_data_path(ip) if ip else None
                if rp and rp.is_file():
                    image_paths.append(str(rp))
        if query_img and query_img.is_file():
            image_paths = [str(query_img)] + image_paths[:2]

        q = f"{question}\n\nLet's think step by step." if use_cot else question
        if cfg["generator"] == "text_llm":
            pred = generator.generate(
                q,
                context_chunks,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        else:
            pred = generator.generate(
                q,
                context_chunks=context_chunks,
                image_paths=image_paths or None,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        rows.append(
            {
                "idx": i,
                "config": config_key,
                "question": question,
                "reference": reference,
                "prediction": pred,
            }
        )

    output_preds.parent.mkdir(parents=True, exist_ok=True)
    final_path = output_preds.parent / f"{output_preds.stem}_{config_key}{output_preds.suffix}"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    elapsed = time.time() - t0
    print(f"Saved {len(rows)} predictions to {final_path}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed / max(len(rows),1):.2f}s/sample)")
    return {"config": config_key, "num_samples": len(rows), "elapsed_seconds": elapsed, "predictions_saved_to": str(final_path)}


def main():
    p = argparse.ArgumentParser(description="Generate predictions only (no metrics).")
    p.add_argument("--config", required=True, choices=sorted(CONFIGS.keys()))
    p.add_argument("--eval-path", type=str, default=None, help="Eval dataset JSON path.")
    p.add_argument("--max-samples", type=int, default=None, help="Use first N eval samples.")
    p.add_argument("--adapter-path", type=str, default=None, help="LoRA adapter path for fine-tuned configs.")
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--alpha", type=float, default=FUSION_ALPHA)
    p.add_argument("--max-new-tokens", type=int, default=128, help="Lower is faster; default 128.")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--use-cot", action="store_true")
    p.add_argument("--output-preds", type=Path, default=Path("data/preds/preds.json"))
    p.add_argument(
        "--b1-backend",
        choices=["local", "api"],
        default="local",
        help="For B1 only: local Mistral or APITextGenerator (needs OPENAI_API_KEY).",
    )
    args = p.parse_args()

    eval_data = load_eval_dataset(Path(args.eval_path) if args.eval_path else None)
    if args.max_samples is not None:
        eval_data = eval_data[: max(0, args.max_samples)]
    run_predictions(
        config_key=args.config,
        eval_data=eval_data,
        output_preds=args.output_preds,
        adapter_path=args.adapter_path,
        top_k=args.top_k,
        alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_cot=args.use_cot,
        b1_backend=args.b1_backend,
    )


if __name__ == "__main__":
    main()
