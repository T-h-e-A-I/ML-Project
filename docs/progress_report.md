# Project Progress Report

**Date:** March 4, 2026
**Project Title:** Multimodal Retrieval-Augmented Generation for STEM Diagram Reasoning
**Base Paper:** *Retrieval-Augmented Generation (RAG) Chatbots for Education: A Survey of Applications* (MDPI, 2025)

---

## 1. What Was Demonstrated in the Last Meeting

In the previous session, we presented **LeVar**, a small contrapositive code-reviewer LLM. The demonstration revealed two critical shortcomings:

- **No dataset:** LeVar did not have a curated training or evaluation dataset, making it impossible to quantify performance or conduct a proper benchmarking study.
- **Outdated foundation:** The project was built on an older paper that did not reflect the current state of the field, limiting the novelty and academic contribution of the work.

These issues made it clear that continuing with LeVar in its current form would not lead to a viable project within the available timeline.

---

## 2. What Was Committed After the Last Discussion

After the last meeting, we committed to one of two directions:

1. **Option A:** Proceed with LeVar and invest the remaining time in building a proper dataset for it.
2. **Option B:** Pivot to a new project idea that aligns better with available resources and timeline constraints.

We chose **Option B** and pivoted to a new project: **Multimodal RAG for STEM Diagrams** -- a benchmarking study comparing how different Vision-Language Models (VLMs) and multimodal embeddings perform on physics diagram reasoning compared to text-only baselines.

**Rationale for the pivot:**
- Stronger novelty: Standard RAG pipelines are text-only and fail on visual content (free-body diagrams, circuit schematics, optics ray diagrams). No comprehensive benchmark exists for multimodal RAG on STEM content.
- Faster execution: Evaluation datasets can be sourced from existing benchmarks (ScienceQA, AI2D) and supplemented synthetically using GPT-4o, eliminating the manual annotation bottleneck that stalled LeVar.
- Clear deep learning component: Fine-tuning a small open-source Vision-Language Model (LLaVA-1.5-7B) using QLoRA on physics diagrams.

---

## 3. Current State of the Project

### 3.1 Codebase Architecture (Complete)

The full project codebase has been implemented with the following modular structure:

```
ML Project/
├── configs/default.py              # Central hyperparameters and model config
├── src/
│   ├── ingestion/                  # Data pipeline
│   │   ├── download_datasets.py    # HuggingFace dataset downloader
│   │   ├── pdf_parser.py           # Textbook PDF parser (PyMuPDF)
│   │   ├── build_knowledge_base.py # End-to-end KB builder
│   │   └── synthetic_generator.py  # GPT-4o synthetic QA generation
│   ├── embeddings/                 # Embedding models
│   │   ├── text_embedder.py        # SentenceTransformer (all-MiniLM-L6-v2)
│   │   └── image_embedder.py       # CLIP (ViT-B/32) via open_clip
│   ├── retrieval/                  # Vector store + retrieval
│   │   ├── vector_store.py         # ChromaDB wrapper
│   │   └── multimodal_retriever.py # Late-fusion multimodal retriever
│   ├── generation/                 # Answer generation
│   │   ├── text_generator.py       # Mistral-7B (4-bit) text-only generator
│   │   ├── vlm_generator.py        # LLaVA-1.5-7B (4-bit) multimodal generator
│   │   └── rag_pipeline.py         # Unified pipeline (all 5 configs)
│   ├── evaluation/                 # Benchmarking
│   │   ├── metrics.py              # Accuracy, ROUGE-L, BERTScore, GPT-4o judge
│   │   ├── run_benchmark.py        # CLI benchmark runner + ablation sweeps
│   │   └── visualize_results.py    # Plots, heatmaps, LaTeX tables
│   └── fine_tuning/                # Model fine-tuning
│       ├── prepare_data.py         # Data formatting for LLaVA
│       └── train_qlora.py          # QLoRA training script
├── notebooks/
│   └── finetune_llava.ipynb        # Kaggle-compatible training notebook
├── paper/
│   └── main.tex                    # IEEE-format paper template
└── docs/
    └── GUIDE.md                    # Full execution guide with verification tests
```

### 3.2 Datasets (Collected and Preprocessed)

| Dataset | Source | Status | Description |
|---|---|---|---|
| ScienceQA (physics subset) | HuggingFace (`derek-thomas/ScienceQA`) | Script ready | Multimodal science QA, filtered by physics keywords |
| AI2D | HuggingFace (`lmms-lab/ai2d`) | Script ready | Annotated science diagrams with questions |
| Synthetic QA | GPT-4o API | Script ready | Physics diagram QA pairs generated from images |
| OpenStax University Physics | OpenStax.org | Pending download | Open-access textbook used as retrieval corpus |

- Download and preprocessing scripts are implemented and tested.
- Physics keyword filtering is configured for 30+ terms (force, momentum, circuit, optics, etc.).
- Unified evaluation set builder merges all sources into a single JSON format.

### 3.3 Version 1: Text-Only RAG Baseline (Complete)

The text-only RAG pipeline (Config B1) is fully implemented:

- **Ingestion:** PDF parser extracts text from textbooks, chunks into 512-token passages with 50-token overlap.
- **Embedding:** Text chunks embedded using SentenceTransformer (`all-MiniLM-L6-v2`).
- **Storage:** ChromaDB persistent vector database.
- **Retrieval:** Top-k cosine similarity search over text embeddings.
- **Generation:** Mistral-7B-Instruct (4-bit quantized) generates answers from retrieved context.
- **Evaluation:** Four metrics implemented -- exact match accuracy, contains accuracy, ROUGE-L, BERTScore. Optional GPT-4o-as-judge for reasoning quality scoring.

### 3.4 Experimental Design (Defined)

Five benchmark configurations have been defined:

| Config | Retrieval | Generator | Fine-tuned | Status |
|---|---|---|---|---|
| B1: Text-only baseline | Text embeddings | Mistral-7B | No | Code complete |
| B2: Text RAG + VLM | Text embeddings | LLaVA-7B | No | Code complete |
| M1: Multimodal RAG | Text + Image embeddings | LLaVA-7B | No | Code complete |
| M2: Multimodal RAG + FT | Text + Image embeddings | LLaVA-7B | Yes (QLoRA) | Code complete |
| M3: No retrieval (VLM only) | None | LLaVA-7B | Yes (QLoRA) | Code complete |

Ablation variables:
- Retrieval depth: k = {3, 5, 10}
- Fusion weight: alpha = {0.3, 0.5, 0.7}
- Chain-of-thought prompting: on/off

---

## 4. Upcoming Tasks / Future To-Dos

### Version 2: Multimodal RAG Pipeline (Next Immediate Priority)

- Download datasets by running the prepared scripts.
- Download and parse OpenStax University Physics textbook PDF.
- Build the ChromaDB knowledge base (text + image embeddings).
- Run Config B1 (text-only baseline) to establish the comparison floor.
- Run Config B2 (text + VLM) and Config M1 (multimodal RAG) to measure the effect of adding image retrieval.
- Record and analyze initial results, identify failure modes on diagram-heavy questions.

### Version 3: Fine-Tuned Multimodal RAG + Paper

- Prepare fine-tuning dataset in LLaVA conversational format (80/10/10 split).
- Fine-tune LLaVA-1.5-7B using QLoRA on Kaggle T4 GPUs (~8-12 hours estimated).
- Run Configs M2 (fine-tuned multimodal RAG) and M3 (fine-tuned VLM without retrieval).
- Execute full ablation study (top-k, fusion alpha, chain-of-thought) -- 11 experiment runs.
- Generate visualization plots (accuracy bar charts, metric heatmaps, ablation line plots).
- Write the research paper using the prepared IEEE-format LaTeX template.
- Final code cleanup, documentation, and submission.

---

## Summary

| Milestone | Status |
|---|---|
| Previous project (LeVar) | Abandoned -- no dataset, outdated base paper |
| New project idea finalized | Done |
| Full codebase architecture | Done (18 source modules) |
| Dataset collection scripts | Done |
| Text-only RAG baseline (V1) | Done (code complete) |
| Evaluation framework | Done (4 metrics + GPT-4o judge) |
| Experimental matrix defined | Done (5 configs + 3 ablation axes) |
| Paper template | Done (IEEE format, 7 sections) |
| Dataset download + preprocessing | Next step |
| Multimodal RAG experiments (V2) | Upcoming |
| QLoRA fine-tuning + ablations (V3) | Upcoming |
| Paper writing + submission | Upcoming |
