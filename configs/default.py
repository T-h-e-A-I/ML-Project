"""Central configuration for the Multimodal RAG pipeline."""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EVAL = PROJECT_ROOT / "data" / "eval"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "data" / "chroma_db"

# ── Text chunking ──────────────────────────────────────────────────────
CHUNK_SIZE = 512          # tokens
CHUNK_OVERLAP = 50        # tokens

# ── Embedding models ──────────────────────────────────────────────────
TEXT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
IMAGE_EMBED_MODEL = "openai/clip-vit-base-patch32"

# ── Generator models ──────────────────────────────────────────────────
TEXT_LLM = "mistralai/Mistral-7B-Instruct-v0.2"
VLM_MODEL = "llava-hf/llava-1.5-7b-hf"

# ── Retrieval ──────────────────────────────────────────────────────────
TOP_K = 5
FUSION_ALPHA = 0.5        # weight for text similarity in late fusion

# ── Fine-tuning (QLoRA) ──────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4

# ── Evaluation ─────────────────────────────────────────────────────────
EVAL_METRICS = ["accuracy", "rouge_l", "bert_score", "gpt4o_judge"]

# ── Collection names ───────────────────────────────────────────────────
TEXT_COLLECTION = "physics_text"
IMAGE_COLLECTION = "physics_images"


def effective_project_root() -> Path:
    """
    Root directory for resolving stored paths (datasets, images).

    Override for another clone or Kaggle layout:

        export ML_PROJECT_ROOT=/path/to/ML-Project

    Paths in JSON are stored relative to this root (POSIX, e.g. data/raw/...).
    """
    return Path(os.environ.get("ML_PROJECT_ROOT", str(PROJECT_ROOT))).expanduser().resolve()


def resolve_data_path(path: str | Path | None) -> Path | None:
    """
    Turn a stored or legacy path into an absolute Path on this machine.

    Accepts project-relative strings (recommended), absolute paths that exist,
    or legacy absolute paths from another machine (remaps by anchoring at the
    first 'data' segment to this repo's layout).
    """
    if path is None:
        return None
    raw = str(path).strip()
    if not raw:
        return None

    root = effective_project_root()
    p = Path(raw)

    if not p.is_absolute():
        return (root / p).resolve()

    if p.exists():
        return p.resolve()

    parts = p.parts
    try:
        idx = parts.index("data")
    except ValueError:
        return p.resolve()

    return root.joinpath(*parts[idx:]).resolve()


def path_for_storage(path: str | Path) -> str:
    """Project-relative POSIX path for JSON / Chroma metadata (portable across machines)."""
    p = Path(path)
    if not p.is_absolute():
        p = (effective_project_root() / p).resolve()
    else:
        p = p.resolve()
    root = effective_project_root()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()
