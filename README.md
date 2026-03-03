# Multimodal RAG for STEM Diagrams

A benchmarking study comparing Vision-Language Models (VLMs) and multimodal embeddings against text-only baselines for physics diagram reasoning, built on a Retrieval-Augmented Generation (RAG) architecture.

## Project Structure

```
ML Project/
├── data/
│   ├── raw/            # Original datasets (ScienceQA, AI2D, textbooks)
│   ├── processed/      # Chunked text, extracted images, formatted data
│   └── eval/           # Evaluation datasets
├── src/
│   ├── ingestion/      # PDF parsing, text+image chunking
│   ├── embeddings/     # Text and image embedding logic
│   ├── retrieval/      # ChromaDB vector store and retrieval
│   ├── generation/     # VLM inference and RAG chain
│   ├── evaluation/     # Benchmarking framework and metrics
│   └── fine_tuning/    # QLoRA fine-tuning scripts for LLaVA
├── notebooks/          # Kaggle-compatible training notebooks
├── configs/            # Hyperparameters and model configs
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Datasets

- **ScienceQA** — Multimodal science questions filtered for physics topics
- **AI2D** — Annotated science diagrams with questions
- **Synthetic** — GPT-4o generated physics QA pairs from diagrams
- **OpenStax** — Open-access University Physics textbook (retrieval corpus)

## Experimental Matrix

| Config | Retrieval | Generator | Fine-tuned |
|---|---|---|---|
| B1 | Text embeddings | Mistral-7B | No |
| B2 | Text embeddings | LLaVA-7B | No |
| M1 | Text + Image embeddings | LLaVA-7B | No |
| M2 | Text + Image embeddings | LLaVA-7B | Yes (QLoRA) |
| M3 | None (direct) | LLaVA-7B | Yes (QLoRA) |

## Reproduction

1. Install dependencies: `pip install -r requirements.txt`
2. Download datasets: `python -m src.ingestion.download_datasets`
3. Build knowledge base: `python -m src.ingestion.build_knowledge_base`
4. Run text-only baseline: `python -m src.evaluation.run_benchmark --config B1`
5. Run multimodal pipeline: `python -m src.evaluation.run_benchmark --config M1`
6. Fine-tune LLaVA: see `notebooks/finetune_llava.ipynb`
7. Run full ablation: `python -m src.evaluation.run_benchmark --all`
