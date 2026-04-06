"""Run benchmark experiments across all configurations."""

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.default import DATA_EVAL, FUSION_ALPHA, TOP_K, resolve_data_path
from src.evaluation.eval_deps import ensure_benchmark_metric_deps
from src.evaluation.metrics import compute_all_metrics

CONFIGS = {
    "B1": {
        "name": "Text-only RAG + Mistral-7B",
        "retrieval": "text",
        "generator": "text_llm",
        "fine_tuned": False,
    },
    "B2": {
        "name": "Text RAG + LLaVA-7B (no fine-tune)",
        "retrieval": "text",
        "generator": "vlm",
        "fine_tuned": False,
    },
    "M1": {
        "name": "Multimodal RAG + LLaVA-7B",
        "retrieval": "multimodal",
        "generator": "vlm",
        "fine_tuned": False,
    },
    "M2": {
        "name": "Multimodal RAG + LLaVA-7B (fine-tuned)",
        "retrieval": "multimodal",
        "generator": "vlm",
        "fine_tuned": True,
    },
    "M3": {
        "name": "No retrieval + LLaVA-7B (fine-tuned)",
        "retrieval": "none",
        "generator": "vlm",
        "fine_tuned": True,
    },
}


def load_eval_dataset(eval_path: Path | None = None) -> list[dict]:
    """Load the unified evaluation dataset."""
    if eval_path is None:
        eval_path = DATA_EVAL / "eval_dataset.json"
    with open(eval_path) as f:
        return json.load(f)


def run_single_config(config_key: str, eval_data: list[dict], **kwargs) -> dict:
    """Run a single benchmark configuration."""
    config = CONFIGS[config_key]
    print(f"\n{'=' * 60}")
    print(f"Running config: {config_key} - {config['name']}")
    print(f"{'=' * 60}")

    retriever = None
    generator = None
    adapter_path = kwargs.get("adapter_path")
    top_k = kwargs.get("top_k", TOP_K)
    alpha = kwargs.get("alpha", FUSION_ALPHA)
    use_cot = kwargs.get("use_cot", False)

    if config["retrieval"] != "none":
        from src.retrieval.multimodal_retriever import MultimodalRetriever
        retriever = MultimodalRetriever()

    if config["generator"] == "text_llm":
        from src.generation.text_generator import TextGenerator
        generator = TextGenerator()
    elif config["generator"] == "vlm":
        from src.generation.vlm_generator import VLMGenerator
        generator = VLMGenerator(
            adapter_path=adapter_path if config["fine_tuned"] else None
        )

    predictions = []
    references = []
    questions = []

    start_time = time.time()

    print(f"Starting evaluation: {len(eval_data)} examples (no log lines until now because models were loading).")
    for sample in tqdm(
        eval_data,
        desc=f"{config_key} eval",
        unit="ex",
        mininterval=2.0,
    ):
        question = sample["question"]
        reference = sample["answer"]
        raw_image = sample.get("image_path")
        query_img = resolve_data_path(raw_image) if raw_image else None

        context_chunks = []
        image_paths = []

        if retriever and config["retrieval"] == "text":
            context_chunks = retriever.retrieve_text_only(question, top_k=top_k)
        elif retriever and config["retrieval"] == "multimodal":
            results = retriever.retrieve_multimodal(
                question,
                query_image_path=str(query_img) if query_img else None,
                top_k=top_k,
                alpha=alpha,
            )
            context_chunks = results["text_results"]
            image_paths = []
            for r in results["image_results"]:
                ip = r.get("image_path")
                rp = resolve_data_path(ip) if ip else None
                if rp and rp.is_file():
                    image_paths.append(str(rp))

        if query_img and query_img.is_file():
            image_paths = [str(query_img)] + image_paths[:2]

        q = question
        if use_cot:
            q = f"{question}\n\nLet's think step by step."

        if config["generator"] == "text_llm":
            pred = generator.generate(q, context_chunks)
        else:
            pred = generator.generate(
                q, context_chunks=context_chunks, image_paths=image_paths or None
            )

        predictions.append(pred)
        references.append(reference)
        questions.append(question)

    elapsed = time.time() - start_time

    predictions_saved_to: str | None = None
    pred_base = kwargs.get("save_predictions_path")
    if pred_base:
        base = Path(pred_base)
        extra = kwargs.get("save_predictions_extra", "")
        path = base.parent / f"{base.stem}_{config_key}{extra}{base.suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"question": q, "reference": r, "prediction": p}
            for q, r, p in zip(questions, references, predictions)
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        predictions_saved_to = str(path.resolve())
        print(f"Saved {len(rows)} predictions to {predictions_saved_to}")

    print("Computing metrics (ROUGE / BERTScore may take several minutes)...")
    metrics = compute_all_metrics(questions, predictions, references)
    metrics["config"] = config_key
    metrics["config_name"] = config["name"]
    metrics["elapsed_seconds"] = elapsed
    metrics["top_k"] = top_k
    metrics["alpha"] = alpha
    if predictions_saved_to:
        metrics["predictions_saved_to"] = predictions_saved_to

    return metrics


def run_all_configs(eval_data: list[dict], **kwargs) -> list[dict]:
    """Run all benchmark configurations."""
    results = []
    for key in CONFIGS:
        if CONFIGS[key]["fine_tuned"] and not kwargs.get("adapter_path"):
            print(f"Skipping {key} (requires adapter_path for fine-tuned model)")
            continue
        result = run_single_config(key, eval_data, **kwargs)
        results.append(result)
        _print_result(result)
    return results


def run_ablations(eval_data: list[dict], **kwargs) -> list[dict]:
    """Run ablation experiments: vary top_k, alpha, and chain-of-thought."""
    results = []

    for top_k in [3, 5, 10]:
        for alpha in [0.3, 0.5, 0.7]:
            ab_kwargs = {
                **kwargs,
                "save_predictions_extra": f"_k{top_k}_a{alpha}",
            }
            result = run_single_config(
                "M1", eval_data, top_k=top_k, alpha=alpha, **ab_kwargs
            )
            results.append(result)
            _print_result(result)

    for use_cot in [False, True]:
        ab_kwargs = {**kwargs, "save_predictions_extra": f"_cot{use_cot}"}
        result = run_single_config(
            "M1", eval_data, use_cot=use_cot, **ab_kwargs
        )
        result["chain_of_thought"] = use_cot
        results.append(result)
        _print_result(result)

    return results


def _print_result(result: dict):
    """Pretty-print a single benchmark result."""
    print(f"\n--- {result['config']} ({result['config_name']}) ---")
    print(f"  Exact match:      {result['exact_match']:.4f}")
    print(f"  Contains accuracy: {result['contains_accuracy']:.4f}")
    print(f"  ROUGE-L F1:        {result['rouge_l']['f1']:.4f}")
    print(f"  BERTScore F1:      {result['bert_score']['f1']:.4f}")
    if "gpt4o_judge" in result:
        print(f"  GPT-4o Correctness:{result['gpt4o_judge'].get('correctness', 'N/A')}")
        print(f"  GPT-4o Reasoning:  {result['gpt4o_judge'].get('reasoning', 'N/A')}")
    print(f"  Time:              {result['elapsed_seconds']:.1f}s")
    print(f"  top_k={result['top_k']}, alpha={result['alpha']}")


def save_results(results: list[dict], output_path: Path | None = None):
    """Save benchmark results to JSON."""
    if output_path is None:
        output_path = DATA_EVAL / "benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for r in results:
        for key in ["rouge_l", "bert_score", "gpt4o_judge"]:
            if key in r and isinstance(r[key], dict):
                for k, v in r[key].items():
                    if hasattr(v, "item"):
                        r[key][k] = float(v)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run multimodal RAG benchmarks")
    parser.add_argument("--config", type=str, default=None,
                        help="Single config to run (B1, B2, M1, M2, M3)")
    parser.add_argument("--all", action="store_true", help="Run all configs")
    parser.add_argument("--ablations", action="store_true", help="Run ablation experiments")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to fine-tuned LoRA adapter")
    parser.add_argument("--eval-path", type=str, default=None,
                        help="Path to evaluation dataset JSON")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--alpha", type=float, default=FUSION_ALPHA)
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results JSON")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Only run on the first N examples (smoke test; default: full eval set)",
    )
    parser.add_argument(
        "--save-predictions",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Write question/reference/prediction JSON before metrics "
            "(so a metrics crash does not lose model outputs). "
            "File is PATH stem + _CONFIG + extension, e.g. preds_M1.json."
        ),
    )
    args = parser.parse_args()

    ensure_benchmark_metric_deps()

    eval_data = load_eval_dataset(Path(args.eval_path) if args.eval_path else None)
    if args.max_samples is not None:
        eval_data = eval_data[: max(0, args.max_samples)]
    print(f"Loaded {len(eval_data)} evaluation samples")

    all_results = []
    pred_path = Path(args.save_predictions) if args.save_predictions else None

    bench_kwargs = dict(
        adapter_path=args.adapter_path,
        top_k=args.top_k,
        alpha=args.alpha,
        save_predictions_path=pred_path,
    )

    if args.config:
        result = run_single_config(args.config, eval_data, **bench_kwargs)
        _print_result(result)
        all_results.append(result)
    elif args.all:
        all_results = run_all_configs(eval_data, **bench_kwargs)
    elif args.ablations:
        all_results = run_ablations(eval_data, **bench_kwargs)
    else:
        parser.print_help()
        return

    save_results(all_results, Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
