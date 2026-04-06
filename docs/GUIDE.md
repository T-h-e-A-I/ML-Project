# Multimodal RAG for STEM Diagrams -- Complete Project Guide

This document covers exactly what was built by the AI assistant, what you (the developer) need to do manually, the exact order of operations, and the tests to verify each step succeeded.

---

## Table of Contents

- [What Was Built (Already Done)](#what-was-built-already-done)
- [What You Need To Do (Manual Steps)](#what-you-need-to-do-manual-steps)
- [Step-by-Step Execution Guide](#step-by-step-execution-guide)
  - [Phase 1: Environment Setup](#phase-1-environment-setup)
  - [Phase 2: Dataset Acquisition](#phase-2-dataset-acquisition)
  - [Phase 3: Knowledge Base Construction](#phase-3-knowledge-base-construction)
  - [Phase 4: Run Baselines (B1, B2)](#phase-4-run-baselines-b1-b2)
  - [Phase 5: Run Multimodal Pipeline (M1)](#phase-5-run-multimodal-pipeline-m1)
  - [Phase 6: Fine-Tuning on Kaggle (M2, M3)](#phase-6-fine-tuning-on-kaggle-m2-m3)
  - [Phase 7: Full Ablation Study](#phase-7-full-ablation-study)
  - [Phase 8: Visualizations and Paper](#phase-8-visualizations-and-paper)
- [File Reference](#file-reference)
- [Troubleshooting](#troubleshooting)

---

## What Was Built (Already Done)

The AI assistant scaffolded the entire codebase. Every Python module, config file, notebook, paper template, and utility is written and ready to run. Nothing needs to be coded from scratch -- you need to provide data and compute.

### Source modules created

| Module | File | Purpose |
|---|---|---|
| Config | `configs/default.py` | All hyperparameters, model names, paths, collection names in one place |
| Dataset download | `src/ingestion/download_datasets.py` | Downloads ScienceQA + AI2D from HuggingFace, filters physics topics, builds unified eval set |
| PDF parser | `src/ingestion/pdf_parser.py` | Extracts text chunks (512 tokens, 50 overlap) and images from textbook PDFs via PyMuPDF |
| Synthetic QA | `src/ingestion/synthetic_generator.py` | Uses GPT-4o API to generate QA pairs from physics diagrams |
| Knowledge base builder | `src/ingestion/build_knowledge_base.py` | Orchestrates PDF parsing, embedding, and ChromaDB storage |
| Text embedder | `src/embeddings/text_embedder.py` | Wraps SentenceTransformer (`all-MiniLM-L6-v2`) |
| Image embedder | `src/embeddings/image_embedder.py` | Wraps CLIP via open_clip (`clip-vit-base-patch32`) |
| Vector store | `src/retrieval/vector_store.py` | ChromaDB wrapper for text + image collections |
| Multimodal retriever | `src/retrieval/multimodal_retriever.py` | Late-fusion retrieval: `score = alpha * text_sim + (1-alpha) * image_sim` |
| Text generator | `src/generation/text_generator.py` | Mistral-7B (4-bit quantized) text-only RAG generator + OpenAI API fallback |
| VLM generator | `src/generation/vlm_generator.py` | LLaVA-1.5-7B (4-bit) multimodal generator with LoRA adapter support |
| RAG pipeline | `src/generation/rag_pipeline.py` | Unified pipeline class supporting all 5 configs (B1/B2/M1/M2/M3) |
| Metrics | `src/evaluation/metrics.py` | Exact match, contains accuracy, ROUGE-L, BERTScore, GPT-4o-as-judge |
| Benchmark runner | `src/evaluation/run_benchmark.py` | CLI tool to run individual configs, all configs, or ablation sweeps |
| Visualizations | `src/evaluation/visualize_results.py` | Bar charts, heatmaps, ablation plots, LaTeX table generator |
| Fine-tune data prep | `src/fine_tuning/prepare_data.py` | Formats data into LLaVA conversational format, creates 80/10/10 splits |
| Fine-tune script | `src/fine_tuning/train_qlora.py` | QLoRA training script (r=16, alpha=32, 4-bit NF4, gradient checkpointing) |
| Kaggle notebook | `notebooks/finetune_llava.ipynb` | Self-contained notebook for Kaggle T4 fine-tuning |
| Paper template | `paper/main.tex` | Full IEEE-format LaTeX template with placeholder tables |

### Non-code files created

| File | Purpose |
|---|---|
| `requirements.txt` | All pip dependencies with minimum versions |
| `README.md` | Project overview and quick-start reproduction steps |
| `.gitignore` | Ignores data, models, caches, LaTeX aux, secrets |
| `paper/main.tex` | Research paper skeleton (Abstract through Conclusion + bibliography) |

---

## What You Need To Do (Manual Steps)

These are things the code cannot do autonomously -- they require your credentials, compute, or manual decisions.

### Must-do items

1. **Install dependencies** -- Run `pip install -r requirements.txt` in a Python 3.10+ environment.
2. **Download datasets** -- Run the download script. Requires internet and ~5 GB disk space.
3. **Obtain a physics textbook PDF** -- Download OpenStax University Physics (free, open-access) and place it in `data/raw/textbooks/`. The PDF parser expects files there.
4. **Set your OpenAI API key** -- Required for synthetic data generation and GPT-4o judge metric. Run `export OPENAI_API_KEY='sk-...'`. If you skip this, synthetic generation and GPT-4o judging are skipped gracefully.
5. **Run benchmarks on GPU** -- Configs B1, B2, M1 require a GPU with at least 16 GB VRAM to load quantized 7B models. Without a local GPU, you can use Kaggle or Colab.
6. **Fine-tune on Kaggle** -- Upload `notebooks/finetune_llava.ipynb` and your processed data to Kaggle. Run with GPU T4 x2 accelerator. Download the adapter weights when done.
7. **Write the paper** -- The LaTeX template in `paper/main.tex` has placeholder values (`0.XXX`). Replace them with your actual benchmark numbers after running experiments.

### Optional items

- **Synthetic data generation** -- If you have an OpenAI API key and want more evaluation data, run the synthetic generator. Cost: ~$5-10 for 300 images.
- **Custom physics diagrams** -- You can add your own physics diagrams to `data/raw/physics_diagrams/` for synthetic QA generation.
- **Modify hyperparameters** -- All tunable values live in `configs/default.py`. Adjust chunk size, top-k, fusion alpha, LoRA rank, etc. as needed.

---

## Step-by-Step Execution Guide

Each phase below lists the exact commands, what they do, and the tests to verify success.

---

### Phase 1: Environment Setup

**Goal:** Working Python environment with all dependencies.

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # On macOS/Linux

# Install all dependencies
pip install -r requirements.txt
```

#### Verification tests

```bash
# Test 1: Core imports work
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Test 2: Transformers + PEFT installed
python -c "from transformers import LlavaForConditionalGeneration; from peft import LoraConfig; print('OK')"

# Test 3: ChromaDB installed
python -c "import chromadb; print(f'ChromaDB {chromadb.__version__}')"

# Test 4: Embedding models loadable
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('all-MiniLM-L6-v2'); print(f'Text embed dim: {m.get_sentence_embedding_dimension()}')"

# Test 5: CLIP loadable
python -c "import open_clip; m, _, p = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai'); print('CLIP OK')"
```

**Pass criteria:** All 5 tests print output without errors. Test 1 shows `CUDA: True` if you have a GPU.

---

### Phase 2: Dataset Acquisition

**Goal:** ScienceQA and AI2D downloaded, physics samples filtered, eval set built.

```bash
# Step 2a: Download and preprocess datasets
python -m src.ingestion.download_datasets
```

This will:
- Download ScienceQA from HuggingFace (~2 GB)
- Filter for physics-related multimodal questions using keyword matching
- Save physics samples to `data/raw/scienceqa/physics_samples.json`
- Save images to `data/raw/scienceqa/images/`
- Download AI2D from HuggingFace
- Save to `data/raw/ai2d/ai2d_samples.json`
- Build unified eval set at `data/eval/eval_dataset.json`

#### Verification tests

```bash
# Test 1: ScienceQA physics samples exist and have content
python -c "
import json
with open('data/raw/scienceqa/physics_samples.json') as f:
    d = json.load(f)
print(f'ScienceQA physics samples: {len(d)}')
assert len(d) > 50, 'Expected at least 50 physics samples'
print('PASS')
"

# Test 2: AI2D samples exist
python -c "
import json
with open('data/raw/ai2d/ai2d_samples.json') as f:
    d = json.load(f)
print(f'AI2D samples: {len(d)}')
assert len(d) > 100, 'Expected at least 100 AI2D samples'
print('PASS')
"

# Test 3: Eval dataset is built
python -c "
import json
with open('data/eval/eval_dataset.json') as f:
    d = json.load(f)
print(f'Eval dataset total: {len(d)}')
sample = d[0]
assert 'question' in sample and 'answer' in sample and 'image_path' in sample
print(f'Sample keys: {list(sample.keys())}')
print('PASS')
"

# Test 4: Images were saved
python -c "
from pathlib import Path
imgs = list(Path('data/raw/scienceqa/images').glob('*.png'))
print(f'ScienceQA images saved: {len(imgs)}')
assert len(imgs) > 0, 'No images found'
print('PASS')
"
```

**Pass criteria:** All 4 tests print `PASS`. You should see at least 50+ ScienceQA physics samples and 100+ AI2D samples.

#### Optional: Synthetic data generation (requires OpenAI API key)

```bash
export OPENAI_API_KEY='sk-your-key-here'
python -m src.ingestion.synthetic_generator
```

Verification:

```bash
python -c "
import json
from pathlib import Path
p = Path('data/eval/synthetic_physics_qa.json')
if p.exists():
    with open(p) as f: d = json.load(f)
    print(f'Synthetic QA pairs: {len(d)}')
    print('PASS')
else:
    print('SKIPPED (no synthetic data generated)')
"
```

---

### Phase 3: Knowledge Base Construction

**Goal:** Physics textbook parsed, text + image embeddings stored in ChromaDB.

#### Prerequisite: Get a textbook PDF

Download OpenStax University Physics (or any open-access physics textbook) and place the PDF file(s) in the `data/raw/textbooks/` directory:

```bash
mkdir -p data/raw/textbooks
# Download OpenStax University Physics Vol 1 (free):
# https://openstax.org/details/books/university-physics-volume-1
# Save the PDF as data/raw/textbooks/university_physics_v1.pdf
```

Then build the knowledge base:

```bash
python -m src.ingestion.build_knowledge_base
```

This will:
1. Parse PDFs with PyMuPDF (extract text + images)
2. Chunk text into 512-token passages with 50-token overlap
3. Embed text chunks with SentenceTransformer
4. Embed images with CLIP
5. Store everything in ChromaDB at `data/chroma_db/`

#### Verification tests

```bash
# Test 1: Text chunks were extracted
python -c "
from pathlib import Path
import json
chunks_files = list(Path('data/processed').rglob('text_chunks.json'))
print(f'Text chunk files found: {len(chunks_files)}')
assert len(chunks_files) > 0, 'No text chunks extracted'
with open(chunks_files[0]) as f:
    chunks = json.load(f)
print(f'Chunks in first file: {len(chunks)}')
print('PASS')
"

# Test 2: Images were extracted
python -c "
from pathlib import Path
import json
img_files = list(Path('data/processed').rglob('images.json'))
print(f'Image metadata files: {len(img_files)}')
if img_files:
    with open(img_files[0]) as f:
        imgs = json.load(f)
    print(f'Images in first file: {len(imgs)}')
print('PASS')
"

# Test 3: ChromaDB has data
python -c "
import sys; sys.path.insert(0, '.')
from src.retrieval.vector_store import VectorStore
store = VectorStore()
text_count = store.get_collection_count('physics_text')
img_count = store.get_collection_count('physics_images')
print(f'ChromaDB text chunks: {text_count}')
print(f'ChromaDB image embeddings: {img_count}')
assert text_count > 0, 'No text embeddings in ChromaDB'
print('PASS')
"
```

**Pass criteria:** All 3 tests print `PASS`. ChromaDB should show > 0 text chunks.

---

### Phase 4: Run Baselines (B1, B2)

**Goal:** Run the text-only RAG baseline and text-RAG + VLM baseline.

**Requirements:** GPU with 16 GB VRAM for local models, OR use OpenAI API fallback for B1.

```bash
# Config B1: Text-only RAG with Mistral-7B
python -m src.evaluation.run_benchmark --config B1

# Config B2: Text RAG with LLaVA-7B (no fine-tuning)
python -m src.evaluation.run_benchmark --config B2
```

Useful flags for long VL runs or Kaggle:

| Flag | Meaning |
|------|--------|
| `--max-samples N` | Only the first N eval examples (smoke test or partial run). |
| `--output PATH` | Metrics JSON path (default: `data/eval/benchmark_results.json`). |
| `--save-predictions PATH` | Writes `question` / `reference` / `prediction` rows **before** metrics. On disk the file is `PATH` with the stem suffixed by `_CONFIG` (e.g. `--save-predictions data/eval/preds.json` → `data/eval/preds_B2.json`). If metrics fail, you still have generations. The metrics JSON also includes `num_samples` and, when used, `predictions_saved_to`. |

Example (200 samples, explicit outputs):

```bash
python -m src.evaluation.run_benchmark --config M1 \
  --max-samples 200 \
  --output data/eval/benchmark_m1_200.json \
  --save-predictions data/eval/preds.json
# Predictions: data/eval/preds_M1.json
```

Each run will:
1. Load the eval dataset
2. For each question: retrieve context from ChromaDB, generate an answer
3. Compute all metrics (exact match, contains accuracy, ROUGE-L, BERTScore)
4. Save results to `data/eval/benchmark_results.json` unless `--output` is set

#### Verification tests

```bash
# Test 1: Results file exists and has B1 data
python -c "
import json
with open('data/eval/benchmark_results.json') as f:
    results = json.load(f)
configs_run = [r['config'] for r in results]
print(f'Configs completed: {configs_run}')
for r in results:
    print(f\"  {r['config']}: exact_match={r.get('exact_match', 'N/A')}, contains_acc={r.get('contains_accuracy', 'N/A')}\")
assert len(results) > 0, 'No results saved'
print('PASS')
"

# Test 2: Metrics are valid numbers
python -c "
import json
with open('data/eval/benchmark_results.json') as f:
    results = json.load(f)
for r in results:
    assert 0 <= r['exact_match'] <= 1, f'Invalid exact_match: {r[\"exact_match\"]}'
    assert 0 <= r['contains_accuracy'] <= 1
    assert 'rouge_l' in r
    assert 'bert_score' in r
print('All metrics valid')
print('PASS')
"
```

**Pass criteria:** Both tests print `PASS`. Results JSON has metric values between 0 and 1.

---

### Phase 5: Run Multimodal Pipeline (M1)

**Goal:** Run multimodal RAG with image retrieval + LLaVA-7B.

```bash
# Config M1: Multimodal RAG + LLaVA-7B (no fine-tuning)
python -m src.evaluation.run_benchmark --config M1
```

#### Verification tests

```bash
# Test: M1 results added to benchmark file
python -c "
import json
with open('data/eval/benchmark_results.json') as f:
    results = json.load(f)
m1 = [r for r in results if r['config'] == 'M1']
assert len(m1) > 0, 'M1 results not found'
print(f'M1 contains_accuracy: {m1[0][\"contains_accuracy\"]}')
print(f'M1 elapsed_seconds: {m1[0][\"elapsed_seconds\"]}')
print('PASS')
"
```

**Pass criteria:** M1 results exist in the JSON. Compare `contains_accuracy` to B1/B2 -- multimodal should ideally outperform text-only on diagram questions.

---

### Phase 6: Fine-Tuning on Kaggle (M2, M3)

**Goal:** Fine-tune LLaVA-1.5-7B with QLoRA on physics data, then run M2 and M3 configs.

#### Step 6a: Prepare fine-tuning data

```bash
python -m src.fine_tuning.prepare_data
```

#### Verification test

```bash
python -c "
import json
from pathlib import Path
for split in ['train', 'val', 'test']:
    p = Path(f'data/processed/finetune/{split}.json')
    assert p.exists(), f'{split}.json not found'
    with open(p) as f:
        d = json.load(f)
    print(f'{split}: {len(d)} samples')
    if d:
        assert 'conversations' in d[0], 'Missing conversations key'
        assert 'image' in d[0], 'Missing image key'
print('PASS')
"
```

#### Step 6b: Fine-tune on Kaggle

1. Go to [kaggle.com](https://www.kaggle.com)
2. Create a new **Dataset** and upload:
   - `data/processed/finetune/train.json`
   - `data/processed/finetune/val.json`
   - The image files referenced in those JSONs (from `data/raw/scienceqa/images/`)
3. Create a new **Notebook**, enable **GPU T4 x2** accelerator
4. Copy the contents of `notebooks/finetune_llava.ipynb` into it (or upload the notebook directly)
5. Update `DATA_DIR` in the notebook to point to your uploaded dataset
6. Run all cells
7. Download the `final_adapter/` folder from the output when training completes

#### Verification test (after downloading adapter)

```bash
# Place downloaded adapter at: outputs/llava_physics_qlora/final_adapter/

python -c "
from pathlib import Path
adapter_dir = Path('outputs/llava_physics_qlora/final_adapter')
assert adapter_dir.exists(), f'Adapter directory not found at {adapter_dir}'
files = list(adapter_dir.iterdir())
file_names = [f.name for f in files]
print(f'Adapter files: {file_names}')
assert any('adapter' in f.lower() or 'lora' in f.lower() or 'safetensors' in f.lower() for f in file_names), 'No adapter weights found'
print('PASS')
"
```

#### Step 6c: Run fine-tuned benchmarks

```bash
# Config M2: Multimodal RAG + fine-tuned LLaVA
python -m src.evaluation.run_benchmark --config M2 \
  --adapter-path outputs/llava_physics_qlora/final_adapter \
  --max-samples 200 \
  --output data/eval/benchmark_m2.json \
  --save-predictions data/eval/preds.json

# Config M3: No retrieval + fine-tuned LLaVA (direct inference)
python -m src.evaluation.run_benchmark --config M3 \
  --adapter-path outputs/llava_physics_qlora/final_adapter \
  --max-samples 200 \
  --output data/eval/benchmark_m3.json \
  --save-predictions data/eval/preds.json
# Predictions land in preds_M2.json / preds_M3.json (shared stem, different suffix)
```

#### Verification test

```bash
python -c "
import json
with open('data/eval/benchmark_results.json') as f:
    results = json.load(f)
configs = {r['config'] for r in results}
print(f'All configs run: {sorted(configs)}')
assert 'M2' in configs, 'M2 not found'
assert 'M3' in configs, 'M3 not found'
m2 = [r for r in results if r['config'] == 'M2'][0]
m1 = [r for r in results if r['config'] == 'M1'][0]
print(f'M1 accuracy: {m1[\"contains_accuracy\"]:.4f}')
print(f'M2 accuracy: {m2[\"contains_accuracy\"]:.4f}')
print(f'Fine-tuning delta: {m2[\"contains_accuracy\"] - m1[\"contains_accuracy\"]:+.4f}')
print('PASS')
"
```

**Pass criteria:** M2 and M3 results appear. M2 should ideally show improvement over M1 (the fine-tuning delta should be positive).

---

### Phase 7: Full Ablation Study

**Goal:** Sweep top-k, fusion alpha, and chain-of-thought prompting.

```bash
# Run all ablation experiments
python -m src.evaluation.run_benchmark --ablations \
  --adapter-path outputs/llava_physics_qlora/final_adapter
```

This will run:
- M1 with top_k = {3, 5, 10} x alpha = {0.3, 0.5, 0.7} (9 combinations)
- M1 with and without chain-of-thought prompting (2 runs)
- Total: 11 additional experiment runs

#### Verification test

```bash
python -c "
import json
with open('data/eval/benchmark_results.json') as f:
    results = json.load(f)
ablation_runs = [r for r in results if r.get('top_k') != 5 or r.get('alpha') != 0.5 or 'chain_of_thought' in r]
print(f'Total results: {len(results)}')
print(f'Ablation runs: {len(ablation_runs)}')
# Check top-k variety
topks = {r['top_k'] for r in results}
alphas = {r['alpha'] for r in results}
print(f'Top-k values tested: {sorted(topks)}')
print(f'Alpha values tested: {sorted(alphas)}')
assert len(topks) >= 3, 'Expected at least 3 different top-k values'
assert len(alphas) >= 3, 'Expected at least 3 different alpha values'
print('PASS')
"
```

**Pass criteria:** At least 3 different top-k and alpha values appear in results.

---

### Phase 8: Visualizations and Paper

**Goal:** Generate plots, tables, and fill in the paper template.

```bash
# Generate all plots and the LaTeX results table
python -m src.evaluation.visualize_results

# If you store per-config files in data/benchmarks, load from there:
python -m src.evaluation.visualize_results --benchmarks-dir data/benchmarks
```

This creates in `data/eval/plots/`:
- `accuracy_comparison.png` -- Bar chart comparing all configs
- `metric_heatmap.png` -- Heatmap of all metrics
- `contains_vs_bertscore.png` -- Trade-off scatter plot
- `runtime_comparison.png` -- Runtime comparison by config
- `ablation_topk.png` -- Line plot of accuracy vs. retrieval depth
- `ablation_alpha.png` -- Line plot of accuracy vs. fusion weight
- `results_table.tex` -- LaTeX table ready to paste into the paper

### Phase 8b: Post-hoc prediction analysis (question-wise)

**Goal:** Analyze saved `data/preds/preds_*.json` without re-running model inference.

```bash
# Generate summary + per-question metrics
python -m src.evaluation.posthoc_analysis --preds-dir data/preds --output-dir data/eval/posthoc
```

Outputs:
- `data/eval/posthoc/posthoc_summary.csv` -- per-config aggregated metrics
- `data/eval/posthoc/question_metrics_<CONFIG>.csv` -- row-level metrics per question
- `data/eval/posthoc/topic_breakdown.csv` -- topic-wise performance breakdown

### Phase 8c: Lightweight local UI for prediction review

```bash
python -m src.evaluation.preds_viewer \
  --preds-dir data/preds \
  --benchmarks-dir data/benchmarks \
  --plots-dir data/eval/plots \
  --posthoc-dir data/eval/posthoc \
  --port 8000
# Open http://127.0.0.1:8000
```

UI features:
- Overview dashboard with benchmark metrics across configs
- Quick per-config prediction stats (hit rate, avg output length)
- Embedded generated plot images (`data/eval/plots/*.png`) when available
- Embedded posthoc summary table (`data/eval/posthoc/posthoc_summary.csv`) when available
- Filter by config, topic, and hit/miss (`contains_ref`)
- Search by question/prediction text
- Browse question-wise reference vs model output
- Side-by-side compare page (`/compare`) with index-based left join (`B1/B2/M1/M2/M3`) and `missing` tags
- Pagination shows `Page X / Y` in both `/questions` and `/compare`

#### Verification test

```bash
python -c "
from pathlib import Path
plot_dir = Path('data/eval/plots')
expected = ['accuracy_comparison.png', 'metric_heatmap.png', 'results_table.tex']
for f in expected:
    p = plot_dir / f
    assert p.exists(), f'Missing: {f}'
    print(f'{f}: {p.stat().st_size / 1024:.1f} KB')
print('PASS')
"
```

#### Filling in the paper

1. Open `paper/main.tex`
2. Replace all `0.XXX` placeholder values with your actual numbers from `data/eval/benchmark_results.json`
3. Copy the generated LaTeX table from `data/eval/plots/results_table.tex` into the Results section
4. Add your name and university
5. Uncomment and add qualitative example figures
6. Compile with `pdflatex main.tex` (or use Overleaf)

---

## File Reference

```
ML Project/
├── configs/
│   ├── __init__.py
│   └── default.py              # All hyperparameters and paths
├── data/
│   ├── raw/                    # YOU PROVIDE: textbook PDFs, raw datasets
│   │   ├── textbooks/          #   Place .pdf files here
│   │   ├── scienceqa/          #   Auto-populated by download script
│   │   └── ai2d/               #   Auto-populated by download script
│   ├── processed/              # Auto-generated: chunks, images, fine-tune data
│   ├── eval/                   # Auto-generated: eval set, results, plots
│   └── chroma_db/              # Auto-generated: vector database
├── docs/
│   └── GUIDE.md                # This file
├── notebooks/
│   └── finetune_llava.ipynb    # Upload to Kaggle for fine-tuning
├── paper/
│   └── main.tex                # Fill in with your results
├── src/
│   ├── ingestion/
│   │   ├── download_datasets.py    # Run first
│   │   ├── pdf_parser.py           # Called by build_knowledge_base
│   │   ├── build_knowledge_base.py # Run after adding textbook PDFs
│   │   └── synthetic_generator.py  # Optional, needs OPENAI_API_KEY
│   ├── embeddings/
│   │   ├── text_embedder.py        # SentenceTransformer wrapper
│   │   └── image_embedder.py       # CLIP wrapper
│   ├── retrieval/
│   │   ├── vector_store.py         # ChromaDB wrapper
│   │   └── multimodal_retriever.py # Late-fusion retrieval
│   ├── generation/
│   │   ├── text_generator.py       # Mistral-7B / OpenAI API
│   │   ├── vlm_generator.py        # LLaVA-1.5-7B
│   │   └── rag_pipeline.py         # Unified pipeline (all configs)
│   ├── evaluation/
│   │   ├── metrics.py              # All evaluation metrics
│   │   ├── run_benchmark.py        # CLI benchmark runner
│   │   └── visualize_results.py    # Plots + LaTeX tables
│   └── fine_tuning/
│       ├── prepare_data.py         # Format data for LLaVA training
│       └── train_qlora.py          # QLoRA training script
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Troubleshooting

### "CUDA out of memory" when loading models

The 7B models require ~5 GB VRAM with 4-bit quantization. If you hit OOM:
- Close other GPU processes
- Reduce `BATCH_SIZE` in `configs/default.py` (try 1 or 2)
- Use the `APITextGenerator` in `text_generator.py` as a fallback (uses OpenAI API, no GPU needed)

### "No text chunks found" during knowledge base build

You need to place at least one PDF in `data/raw/textbooks/`. Download OpenStax University Physics from https://openstax.org/details/books/university-physics-volume-1.

### HuggingFace download fails or is slow

Some datasets are gated. Make sure you have `huggingface-hub` installed and are logged in:
```bash
huggingface-cli login
```

### Fine-tuning runs out of time on Kaggle

Kaggle gives 30 hours/week of GPU time. The fine-tuning is estimated at 8-12 hours. If it exceeds this:
- Reduce `NUM_EPOCHS` from 3 to 2 in the notebook
- Reduce training data size (take a subset of `train.json`)
- The notebook saves checkpoints every 200 steps, so you can resume from the last checkpoint

### Synthetic generator says "OPENAI_API_KEY not set"

This is expected if you haven't set the key. Synthetic data generation is **optional**. The pipeline works fine with just ScienceQA + AI2D data. If you want synthetic data:
```bash
export OPENAI_API_KEY='sk-your-key-here'
python -m src.ingestion.synthetic_generator
```

### BERTScore is slow

BERTScore downloads a BERT model on first run (~400 MB). Subsequent runs are cached. If it's too slow, you can disable it by editing `src/evaluation/metrics.py` and removing `bert_score` from the `compute_all_metrics` function.

---

## Quick Command Summary

```bash
# -- SETUP --
pip install -r requirements.txt

# -- DATA --
python -m src.ingestion.download_datasets
# (place textbook PDFs in data/raw/textbooks/)
python -m src.ingestion.build_knowledge_base

# -- OPTIONAL: synthetic data --
export OPENAI_API_KEY='sk-...'
python -m src.ingestion.synthetic_generator

# -- BENCHMARKS --
python -m src.evaluation.run_benchmark --config B1
python -m src.evaluation.run_benchmark --config B2
python -m src.evaluation.run_benchmark --config M1

# -- FINE-TUNING --
python -m src.fine_tuning.prepare_data
# (upload to Kaggle, run notebook, download adapter)
python -m src.evaluation.run_benchmark --config M2 --adapter-path outputs/llava_physics_qlora/final_adapter \
  --max-samples 200 --output data/eval/benchmark_m2.json --save-predictions data/eval/preds.json
python -m src.evaluation.run_benchmark --config M3 --adapter-path outputs/llava_physics_qlora/final_adapter \
  --max-samples 200 --output data/eval/benchmark_m3.json --save-predictions data/eval/preds.json

# Kaggle example paths: --output /kaggle/working/benchmark_m2.json --save-predictions /kaggle/working/preds.json

# -- ABLATIONS --
python -m src.evaluation.run_benchmark --ablations --adapter-path outputs/llava_physics_qlora/final_adapter

# -- PLOTS + PAPER --
python -m src.evaluation.visualize_results
```
