"""Central configuration for the Multimodal RAG pipeline."""

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
