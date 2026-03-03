"""Visualization utilities for benchmark results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from configs.default import DATA_EVAL


def load_results(results_path: Path | None = None) -> list[dict]:
    """Load benchmark results from JSON."""
    if results_path is None:
        results_path = DATA_EVAL / "benchmark_results.json"
    with open(results_path) as f:
        return json.load(f)


def plot_accuracy_comparison(results: list[dict], output_path: Path | None = None):
    """Bar chart comparing accuracy metrics across configurations."""
    configs = [r["config"] for r in results]
    exact_match = [r.get("exact_match", 0) for r in results]
    contains_acc = [r.get("contains_accuracy", 0) for r in results]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, exact_match, width, label="Exact Match", color="#2196F3")
    bars2 = ax.bar(x + width / 2, contains_acc, width, label="Contains Accuracy", color="#FF9800")

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison Across Configurations")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{r.get('config_name', '')[:25]}" for c, r in zip(configs, results)],
                       fontsize=8, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.0)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved accuracy plot to {output_path}")
    plt.show()


def plot_metric_heatmap(results: list[dict], output_path: Path | None = None):
    """Heatmap of all metrics across configurations."""
    rows = []
    for r in results:
        row = {
            "Config": r["config"],
            "Exact Match": r.get("exact_match", 0),
            "Contains Acc": r.get("contains_accuracy", 0),
            "ROUGE-L F1": r.get("rouge_l", {}).get("f1", 0),
            "BERTScore F1": r.get("bert_score", {}).get("f1", 0),
        }
        if "gpt4o_judge" in r:
            row["GPT-4o Correct"] = r["gpt4o_judge"].get("correctness", 0) / 5.0
            row["GPT-4o Reason"] = r["gpt4o_judge"].get("reasoning", 0) / 5.0
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Config")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                vmin=0, vmax=1, linewidths=0.5)
    ax.set_title("Metric Comparison Heatmap")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap to {output_path}")
    plt.show()


def plot_ablation_topk(results: list[dict], output_path: Path | None = None):
    """Line plot of accuracy vs top_k for ablation study."""
    topk_results = {}
    for r in results:
        k = r.get("top_k", 5)
        if k not in topk_results:
            topk_results[k] = []
        topk_results[k].append(r.get("contains_accuracy", 0))

    ks = sorted(topk_results.keys())
    accs = [np.mean(topk_results[k]) for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, accs, "o-", color="#4CAF50", linewidth=2, markersize=8)
    ax.set_xlabel("Top-K Retrieval Depth")
    ax.set_ylabel("Contains Accuracy")
    ax.set_title("Effect of Retrieval Depth on Accuracy")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved top-k plot to {output_path}")
    plt.show()


def plot_ablation_alpha(results: list[dict], output_path: Path | None = None):
    """Line plot of accuracy vs fusion alpha for ablation study."""
    alpha_results = {}
    for r in results:
        a = r.get("alpha", 0.5)
        if a not in alpha_results:
            alpha_results[a] = []
        alpha_results[a].append(r.get("contains_accuracy", 0))

    alphas = sorted(alpha_results.keys())
    accs = [np.mean(alpha_results[a]) for a in alphas]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, accs, "s-", color="#9C27B0", linewidth=2, markersize=8)
    ax.set_xlabel("Fusion Alpha (text weight)")
    ax.set_ylabel("Contains Accuracy")
    ax.set_title("Effect of Fusion Weight on Accuracy")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved alpha plot to {output_path}")
    plt.show()


def generate_latex_table(results: list[dict]) -> str:
    """Generate a LaTeX table of results for the paper."""
    header = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Benchmark Results Across Configurations}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Config & Exact Match & Contains Acc & ROUGE-L F1 & BERTScore F1 \\\\\n"
        "\\midrule\n"
    )

    rows = []
    for r in results:
        rows.append(
            f"{r['config']} & "
            f"{r.get('exact_match', 0):.3f} & "
            f"{r.get('contains_accuracy', 0):.3f} & "
            f"{r.get('rouge_l', {}).get('f1', 0):.3f} & "
            f"{r.get('bert_score', {}).get('f1', 0):.3f} \\\\"
        )

    footer = (
        "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\label{tab:results}\n"
        "\\end{table}"
    )

    return header + "\n".join(rows) + footer


def generate_all_plots(results_path: Path | None = None, output_dir: Path | None = None):
    """Generate all visualization plots."""
    results = load_results(results_path)

    if output_dir is None:
        output_dir = DATA_EVAL / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_comparison(results, output_dir / "accuracy_comparison.png")
    plot_metric_heatmap(results, output_dir / "metric_heatmap.png")

    ablation_results = [r for r in results if r.get("top_k") != 5 or r.get("alpha") != 0.5]
    if ablation_results:
        plot_ablation_topk(results, output_dir / "ablation_topk.png")
        plot_ablation_alpha(results, output_dir / "ablation_alpha.png")

    latex = generate_latex_table(results)
    with open(output_dir / "results_table.tex", "w") as f:
        f.write(latex)
    print(f"Saved LaTeX table to {output_dir / 'results_table.tex'}")


if __name__ == "__main__":
    generate_all_plots()
