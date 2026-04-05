"""Fail fast when benchmark metric dependencies are missing (avoid long runs then ImportError)."""

from __future__ import annotations

# Import name -> pip install name
_BENCHMARK_METRIC_PACKAGES: tuple[tuple[str, str], ...] = (
    ("rouge_score", "rouge-score"),
    ("bert_score", "bert-score"),
)


def ensure_benchmark_metric_deps() -> None:
    """
    Import ROUGE and BERTScore deps before loading models or running retrieval.

    Raises SystemExit with install hints if anything is missing.
    """
    missing: list[tuple[str, str]] = []
    for import_name, pip_name in _BENCHMARK_METRIC_PACKAGES:
        try:
            __import__(import_name)
        except ImportError:
            missing.append((import_name, pip_name))
    if not missing:
        return

    pip_list = " ".join(pip for _, pip in missing)
    msg = (
        "Missing package(s) required for evaluation metrics:\n"
        + "\n".join(f"  - {pip}  (Python: import {imp})" for imp, pip in missing)
        + "\n\nInstall:\n"
        f"  pip install {pip_list}\n\n"
        "Or from the project root:\n"
        "  pip install -r requirements-colab.txt\n"
        "  pip install -r requirements.txt\n"
    )
    raise SystemExit(msg)
