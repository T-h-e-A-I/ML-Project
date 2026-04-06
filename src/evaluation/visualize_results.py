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


def _load_posthoc_frames(posthoc_dir: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Load posthoc CSV artifacts if present."""
    summary = None
    topic = None
    qmetrics = None
    summary_path = posthoc_dir / "posthoc_summary.csv"
    topic_path = posthoc_dir / "topic_breakdown.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
    if topic_path.exists():
        topic = pd.read_csv(topic_path)
    q_paths = sorted(posthoc_dir.glob("question_metrics_*.csv"))
    if q_paths:
        qmetrics = pd.concat([pd.read_csv(p) for p in q_paths], ignore_index=True)
    return summary, topic, qmetrics


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


def plot_quality_profile(results: list[dict], output_path: Path | None = None):
    """Radar-style normalized profile across key metrics."""
    df = _to_plot_df(results).copy()
    dims = ["Contains Acc", "ROUGE-L F1", "BERTScore F1", "GPT Correct", "GPT Reason"]
    # Normalize GPT metrics to [0,1]
    df["GPT Correct"] = df["GPT Correct"] / 5.0
    df["GPT Reason"] = df["GPT Reason"] / 5.0

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(8.2, 8.2))
    ax = plt.subplot(111, polar=True)

    for _, row in df.iterrows():
        vals = [float(row[d]) if pd.notna(row[d]) else 0.0 for d in dims]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=row["Config"])
        ax.fill(angles, vals, alpha=0.08)

    ax.set_thetagrids(np.degrees(angles[:-1]), ["Contains", "ROUGE-L", "BERTScore", "GPT Corr/5", "GPT Reas/5"])
    ax.set_ylim(0, 1)
    ax.set_title("Configuration Quality Profile", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    _savefig(fig, output_path, "quality profile")


def plot_efficiency_frontier(results: list[dict], output_path: Path | None = None):
    """Quality vs runtime frontier view."""
    df = _to_plot_df(results).copy()
    df["Elapsed Min"] = df["Elapsed Sec"] / 60.0
    # Composite quality (simple equal weighting).
    df["Quality"] = (
        0.35 * df["Contains Acc"]
        + 0.25 * df["BERTScore F1"]
        + 0.2 * (df["GPT Correct"] / 5.0).fillna(0)
        + 0.2 * (df["GPT Reason"] / 5.0).fillna(0)
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.scatter(df["Elapsed Min"], df["Quality"], s=110, c="#0ea5e9")
    for _, row in df.iterrows():
        ax.annotate(row["Config"], (row["Elapsed Min"] + 0.2, row["Quality"] + 0.003), fontsize=9)
    ax.set_xlabel("Runtime (minutes)")
    ax.set_ylabel("Composite Quality (0-1)")
    ax.set_title("Efficiency Frontier: Quality vs Runtime")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _savefig(fig, output_path, "efficiency frontier")


def plot_delta_vs_m1(results: list[dict], output_path: Path | None = None):
    """Show metric deltas relative to M1 baseline."""
    df = _to_plot_df(results).set_index("Config")
    if "M1" not in df.index:
        return
    base = df.loc["M1"]
    metrics = ["Contains Acc", "ROUGE-L F1", "BERTScore F1"]
    rows = []
    for cfg, row in df.iterrows():
        for m in metrics:
            rows.append({"Config": cfg, "Metric": m, "Delta": float(row[m] - base[m])})
    ddf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    sns.barplot(ddf, x="Config", y="Delta", hue="Metric", ax=ax)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_title("Metric Delta vs M1 Baseline")
    ax.set_ylabel("Delta")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _savefig(fig, output_path, "delta vs M1")


def plot_metric_rankings(results: list[dict], output_path: Path | None = None):
    """Rank each config across key metrics (lower rank is better)."""
    df = _to_plot_df(results).copy()
    metrics = {
        "Contains": "Contains Acc",
        "ROUGE-L": "ROUGE-L F1",
        "BERTScore": "BERTScore F1",
        "GPT Corr": "GPT Correct",
        "GPT Reas": "GPT Reason",
    }
    ranks = {"Config": df["Config"].tolist()}
    for name, col in metrics.items():
        ranks[name] = df[col].rank(ascending=False, method="dense").tolist()
    rdf = pd.DataFrame(ranks).set_index("Config")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.heatmap(rdf, annot=True, fmt=".0f", cmap="Blues_r", vmin=1, vmax=max(1, len(df)), ax=ax, linewidths=0.4)
    ax.set_title("Rankings by Metric (1 = best)")
    fig.tight_layout()
    _savefig(fig, output_path, "metric rankings")


def plot_posthoc_length_vs_contains(summary_df: pd.DataFrame, output_path: Path | None = None):
    """Scatter: average output length vs contains accuracy (posthoc)."""
    if summary_df is None or summary_df.empty:
        return
    required = {"config", "avg_pred_words", "contains_ref_acc"}
    if not required.issubset(summary_df.columns):
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(summary_df["avg_pred_words"], summary_df["contains_ref_acc"], s=100, c="#14b8a6")
    for _, row in summary_df.iterrows():
        ax.annotate(row["config"], (row["avg_pred_words"] + 2, row["contains_ref_acc"] + 0.002), fontsize=9)
    ax.set_xlabel("Average Prediction Length (words)")
    ax.set_ylabel("Contains Accuracy (posthoc)")
    ax.set_title("Length vs Contains (Posthoc)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _savefig(fig, output_path, "posthoc length-vs-contains")


def plot_posthoc_repetition(summary_df: pd.DataFrame, output_path: Path | None = None):
    """Bar: repetition rate by config (posthoc)."""
    if summary_df is None or summary_df.empty:
        return
    required = {"config", "repetition_rate"}
    if not required.issubset(summary_df.columns):
        return
    df = summary_df.copy().sort_values("config")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(df["config"], df["repetition_rate"], color="#f97316")
    ax.set_ylabel("Repetition Rate")
    ax.set_title("Output Repetition by Configuration (Posthoc)")
    ax.set_ylim(0, max(0.1, float(df["repetition_rate"].max()) + 0.05))
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.005, f"{h:.3f}", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _savefig(fig, output_path, "posthoc repetition")


def plot_posthoc_first_sentence_delta(summary_df: pd.DataFrame, output_path: Path | None = None):
    """Delta between first-sentence and full-answer metrics."""
    if summary_df is None or summary_df.empty:
        return
    req = {
        "config",
        "token_f1_mean",
        "token_f1_first_sentence_mean",
        "rougeL_f1_mean",
        "rougeL_f1_first_sentence_mean",
    }
    if not req.issubset(summary_df.columns):
        return
    rows = []
    for _, r in summary_df.iterrows():
        rows.append(
            {
                "Config": r["config"],
                "Metric": "Token F1 Δ(first-full)",
                "Delta": float(r["token_f1_first_sentence_mean"] - r["token_f1_mean"]),
            }
        )
        rows.append(
            {
                "Config": r["config"],
                "Metric": "ROUGE-L Δ(first-full)",
                "Delta": float(r["rougeL_f1_first_sentence_mean"] - r["rougeL_f1_mean"]),
            }
        )
    ddf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(ddf, x="Config", y="Delta", hue="Metric", ax=ax)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_title("First Sentence vs Full Answer Delta (Posthoc)")
    ax.set_ylabel("Delta")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _savefig(fig, output_path, "posthoc first-sentence delta")


def plot_posthoc_topic_heatmap(topic_df: pd.DataFrame, output_path: Path | None = None):
    """Heatmap of topic-wise contains accuracy."""
    if topic_df is None or topic_df.empty:
        return
    req = {"config", "topic", "contains_ref"}
    if not req.issubset(topic_df.columns):
        return
    pivot = topic_df.pivot(index="config", columns="topic", values="contains_ref").fillna(0)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, linewidths=0.4, ax=ax)
    ax.set_title("Topic-wise Contains Accuracy (Posthoc)")
    fig.tight_layout()
    _savefig(fig, output_path, "posthoc topic heatmap")


def plot_posthoc_word_boxplot(qmetrics_df: pd.DataFrame, output_path: Path | None = None):
    """Distribution of prediction lengths by config."""
    if qmetrics_df is None or qmetrics_df.empty:
        return
    req = {"config", "pred_words"}
    if not req.issubset(qmetrics_df.columns):
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(qmetrics_df, x="config", y="pred_words", ax=ax)
    ax.set_title("Prediction Length Distribution by Config (Posthoc)")
    ax.set_ylabel("Words per prediction")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _savefig(fig, output_path, "posthoc word distribution")


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
    posthoc_dir: Path | None = None,
    output_dir: Path | None = None,
):
    """Generate all plots and table artifacts."""
    results = load_results(results_path=results_path, benchmarks_dir=benchmarks_dir)
    if not results:
        raise ValueError("No benchmark rows found.")

    if output_dir is None:
        output_dir = DATA_EVAL / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    if posthoc_dir is None:
        posthoc_dir = DATA_EVAL / "posthoc"

    plot_accuracy_comparison(results, output_dir / "accuracy_comparison.png")
    plot_metric_heatmap(results, output_dir / "metric_heatmap.png")
    plot_tradeoff_scatter(results, output_dir / "contains_vs_bertscore.png")
    plot_runtime(results, output_dir / "runtime_comparison.png")
    plot_quality_profile(results, output_dir / "quality_profile_radar.png")
    plot_efficiency_frontier(results, output_dir / "efficiency_frontier.png")
    plot_delta_vs_m1(results, output_dir / "delta_vs_m1.png")
    plot_metric_rankings(results, output_dir / "metric_rankings.png")
    plot_ablation_topk(results, output_dir / "ablation_topk.png")
    plot_ablation_alpha(results, output_dir / "ablation_alpha.png")

    # Optional posthoc plots
    summary_df, topic_df, qmetrics_df = _load_posthoc_frames(posthoc_dir)
    plot_posthoc_length_vs_contains(summary_df, output_dir / "posthoc_length_vs_contains.png")
    plot_posthoc_repetition(summary_df, output_dir / "posthoc_repetition_rate.png")
    plot_posthoc_first_sentence_delta(summary_df, output_dir / "posthoc_first_sentence_delta.png")
    plot_posthoc_topic_heatmap(topic_df, output_dir / "posthoc_topic_contains_heatmap.png")
    plot_posthoc_word_boxplot(qmetrics_df, output_dir / "posthoc_word_distribution.png")

    latex = generate_latex_table(results)
    latex_path = output_dir / "results_table.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Saved LaTeX table to {latex_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark visualizations for paper.")
    parser.add_argument("--results-path", type=Path, default=None, help="Path to combined benchmark_results.json.")
    parser.add_argument("--benchmarks-dir", type=Path, default=None, help="Directory with benchmark_*.json files.")
    parser.add_argument("--posthoc-dir", type=Path, default=DATA_EVAL / "posthoc", help="Directory with posthoc CSV artifacts.")
    parser.add_argument("--output-dir", type=Path, default=DATA_EVAL / "plots", help="Output directory for plots.")
    args = parser.parse_args()
    generate_all_plots(
        results_path=args.results_path,
        benchmarks_dir=args.benchmarks_dir,
        posthoc_dir=args.posthoc_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
