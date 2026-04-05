"""Fail fast when QLoRA fine-tuning dependencies are missing (before data/model load)."""

from __future__ import annotations

# Import name -> pip install name
_QLORA_TRAIN_PACKAGES: tuple[tuple[str, str], ...] = (
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("peft", "peft"),
    ("bitsandbytes", "bitsandbytes"),
    ("accelerate", "accelerate"),
    ("datasets", "datasets"),
    ("PIL", "Pillow"),
    ("tqdm", "tqdm"),
    ("sentencepiece", "sentencepiece"),
)


def ensure_qlora_train_deps() -> None:
    """
    Verify imports needed for train_qlora (especially bitsandbytes & accelerate,
    which are not imported at module top level and otherwise fail mid-run).

    Raises SystemExit with install hints if anything is missing.
    """
    missing: list[tuple[str, str]] = []
    for import_name, pip_name in _QLORA_TRAIN_PACKAGES:
        try:
            __import__(import_name)
        except ImportError:
            missing.append((import_name, pip_name))
    if not missing:
        return

    pip_list = " ".join(p for _, p in missing)
    msg = (
        "Missing package(s) required for QLoRA fine-tuning (train_qlora):\n"
        + "\n".join(f"  - {pip}  (Python: import {imp})" for imp, pip in missing)
        + "\n\nInstall:\n"
        f"  pip install {pip_list}\n\n"
        "Or from the project root:\n"
        "  pip install -r requirements-colab.txt\n"
        "  pip install -r requirements.txt\n"
    )
    raise SystemExit(msg)
