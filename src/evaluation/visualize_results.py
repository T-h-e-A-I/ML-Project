"""Visualization utilities for benchmark results and paper figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from configs.default import DATA_EVAL

CANONICAL_ORDER = ["B1", "B2", "M1", "M2", "M3"]


def _load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON format at {path}")


def load_results(
    results_path: Path | None = None,
    benchmarks_dir: Path | None = None,
) -> list[dict]:
    """
    Load benchmark results from either:
    - a combined benchmark_results.json (results_path), or
    - a directory containing benchmark_*.json (benchmarks_dir).
    """
    if results_path is None and benchmarks_dir is None:
        default_results = DATA_EVAL / "benchmark_results.json"
        default_benchmarks = DATA_EVAL.parent / "benchmarks"
        if default_results.exists():
            results_path = default_results
        elif default_benchmarks.exists():
            benchmarks_dir = default_benchmarks
        else:
            raise FileNotFoundError(
                "No results found. Pass --results-path or --benchmarks-dir."
            )

    rows: list[dict] = []
    if results_path is not None:
        rows.extend(_load_json(results_path))
    if benchmarks_dir is not None:
        for p in sorted(benchmarks_dir.glob("benchmark_*.json")):
            rows.extend(_load_json(p))

    dedup: dict[str, dict] = {}
    for r in rows:
        cfg = r.get("config")
        if cfg:
            dedup[cfg] = r
    ordered = sorted(
        dedup.values(),
        key=lambda x: (
            CANONICAL_ORDER.index(x["config"])
            if x.get("config") in CANONICAL_ORDER
            else 999,
            x.get("config", ""),
        ),
    )
    return ordered


def _savefig(fig, output_path: Path | None, label: str):
    if output_path:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved {label} to {output_path}")
    plt.close(fig)


def _to_plot_df(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "Config": r.get("config", "N/A"),
                "Name": r.get("config_name", ""),
                "Exact Match": r.get("exact_match", 0.0),
                "Contains Acc": r.get("contains_accuracy", 0.0),
                "ROUGE-L F1": r.get("rouge_l", {}).get("f1", 0.0),
                "BERTScore F1": r.get("bert_score", {}).get("f1", 0.0),
                "GPT Correct": r.get("gpt4o_judge", {}).get("correctness", np.nan),
                "GPT Reason": r.get("gpt4o_judge", {}).get("reasoning", np.nan),
                "Elapsed Sec": r.get("elapsed_seconds", np.nan),
                "Samples": r.get("num_samples", np.nan),
            }
        )
    return pd.DataFrame(rows)


def plot_accuracy_comparison(results: list[dict], output_path: Path | None = None):
    """Bar chart comparing exact and contains accuracy."""
    df = _to_plot_df(results)
    x = np.arange(len(df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width / 2, df["Exact Match"], width, label="Exact", color="#1f77b4")
    b2 = ax.bar(x + width / 2, df["Contains Acc"], width, label="Contains", color="#ff7f0e")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Accuracy Metrics by Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Config"])
    ax.legend()

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    _savefig(fig, output_path, "accuracy plot")


def plot_metric_heatmap(results: list[dict], output_path: Path | None = None):
    """Paper-style heatmap of normalized metrics."""
    df = _to_plot_df(results).set_index("Config")
    matrix = pd.DataFrame(
        {
            "Exact": df["Exact Match"],
            "Contains": df["Contains Acc"],
            "ROUGE-L": df["ROUGE-L F1"],
            "BERTScore": df["BERTScore F1"],
            "GPT Corr/5": df["GPT Correct"] / 5.0,
            "GPT Reas/5": df["GPT Reason"] / 5.0,
        }
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=1, linewidths=0.4, ax=ax)
    ax.set_title("Normalized Metric Heatmap")
    fig.tight_layout()
    _savefig(fig, output_path, "metric heatmap")


def plot_tradeoff_scatter(results: list[dict], output_path: Path | None = None):
    """Contains-vs-quality scatter to visualize trade-offs."""
    df = _to_plot_df(results)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(df["Contains Acc"], df["BERTScore F1"], s=90, c="#2ca02c")
    for _, row in df.iterrows():
        ax.annotate(row["Config"], (row["Contains Acc"] + 0.003, row["BERTScore F1"] + 0.001), fontsize=9)
    ax.set_xlim(0, min(1.0, max(0.25, df["Contains Acc"].max() + 0.03)))
    ax.set_ylim(0.7, 0.9)
    ax.set_xlabel("Contains Accuracy")
    ax.set_ylabel("BERTScore F1")
    ax.set_title("Contains vs Semantic Overlap Trade-off")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _savefig(fig, output_path, "tradeoff scatter")


def plot_runtime(results: list[dict], output_path: Path | None = None):
    """Runtime comparison in minutes."""
    df = _to_plot_df(results).copy()
    df["Elapsed Min"] = df["Elapsed Sec"] / 60.0
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(df["Config"], df["Elapsed Min"], color="#9467bd")
    ax.set_ylabel("Minutes")
    ax.set_title("Runtime by Configuration")
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.6, f"{h:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    _savefig(fig, output_path, "runtime plot")


def plot_ablation_topk(results: list[dict], output_path: Path | None = None):
    """Line plot of contains accuracy vs top-k (ablation-only rows)."""
    ablation = [r for r in results if r.get("top_k") != 5 or r.get("alpha") != 0.5]
    if not ablation:
        return
    grouped: dict[int, list[float]] = {}
    for r in ablation:
        k = int(r.get("top_k", 5))
        grouped.setdefault(k, []).append(float(r.get("contains_accuracy", 0)))
    ks = sorted(grouped)
    vals = [float(np.mean(grouped[k])) for k in ks]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, vals, "o-", color="#4CAF50", linewidth=2)
    ax.set_xlabel("Top-k")
    ax.set_ylabel("Contains Accuracy")
    ax.set_title("Ablation: Retrieval Depth")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _savefig(fig, output_path, "ablation top-k plot")


def plot_ablation_alpha(results: list[dict], output_path: Path | None = None):
    """Line plot of contains accuracy vs alpha (ablation-only rows)."""
    ablation = [r for r in results if r.get("top_k") != 5 or r.get("alpha") != 0.5]
    if not ablation:
        return
    grouped: dict[float, list[float]] = {}
    for r in ablation:
        a = float(r.get("alpha", 0.5))
        grouped.setdefault(a, []).append(float(r.get("contains_accuracy", 0)))
    alphas = sorted(grouped)
    vals = [float(np.mean(grouped[a])) for a in alphas]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(alphas, vals, "s-", color="#9C27B0", linewidth=2)
    ax.set_xlabel("Fusion alpha")
    ax.set_ylabel("Contains Accuracy")
    ax.set_title("Ablation: Fusion Weight")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _savefig(fig, output_path, "ablation alpha plot")


def generate_latex_table(results: list[dict]) -> str:
    """Generate a LaTeX results table for paper inclusion."""
    header = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Benchmark results across configurations (200-sample runs where available).}\n"
        "\\begin{tabular}{lcccccc}\n"
        "\\toprule\n"
        "Config & Contains & ROUGE-L & BERTScore & GPT Corr. & GPT Reas. & Time (min) \\\\\n"
        "\\midrule\n"
    )
    lines = []
    for r in results:
        g = r.get("gpt4o_judge", {})
        lines.append(
            f"{r.get('config', 'N/A')} & "
            f"{r.get('contains_accuracy', 0):.3f} & "
            f"{r.get('rouge_l', {}).get('f1', 0):.3f} & "
            f"{r.get('bert_score', {}).get('f1', 0):.3f} & "
            f"{g.get('correctness', 0):.2f} & "
            f"{g.get('reasoning', 0):.2f} & "
            f"{(r.get('elapsed_seconds', 0) / 60.0):.1f} \\\\"
        )
    footer = (
        "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\label{tab:benchmarks}\n"
        "\\end{table}\n"
    )
    return header + "\n".join(lines) + footer


def generate_all_plots(
    results_path: Path | None = None,
    benchmarks_dir: Path | None = None,
    output_dir: Path | None = None,
):
    """Generate all plots and table artifacts."""
    results = load_results(results_path=results_path, benchmarks_dir=benchmarks_dir)
    if not results:
        raise ValueError("No benchmark rows found.")

    if output_dir is None:
        output_dir = DATA_EVAL / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_comparison(results, output_dir / "accuracy_comparison.png")
    plot_metric_heatmap(results, output_dir / "metric_heatmap.png")
    plot_tradeoff_scatter(results, output_dir / "contains_vs_bertscore.png")
    plot_runtime(results, output_dir / "runtime_comparison.png")
    plot_ablation_topk(results, output_dir / "ablation_topk.png")
    plot_ablation_alpha(results, output_dir / "ablation_alpha.png")

    latex = generate_latex_table(results)
    latex_path = output_dir / "results_table.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Saved LaTeX table to {latex_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark visualizations for paper.")
    parser.add_argument("--results-path", type=Path, default=None, help="Path to combined benchmark_results.json.")
    parser.add_argument("--benchmarks-dir", type=Path, default=None, help="Directory with benchmark_*.json files.")
    parser.add_argument("--output-dir", type=Path, default=DATA_EVAL / "plots", help="Output directory for plots.")
    args = parser.parse_args()
    generate_all_plots(
        results_path=args.results_path,
        benchmarks_dir=args.benchmarks_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
