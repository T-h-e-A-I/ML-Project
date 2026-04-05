"""Orchestrate the full knowledge base build: parse PDFs, embed, store in ChromaDB."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.ingestion.pdf_parser import build_knowledge_base as parse_pdfs
from src.embeddings.text_embedder import TextEmbedder
from src.embeddings.image_embedder import ImageEmbedder
from src.retrieval.vector_store import VectorStore
from configs.default import (
    DATA_PROCESSED,
    IMAGE_COLLECTION,
    TEXT_COLLECTION,
    path_for_storage,
    resolve_data_path,
)


def build():
    """Full pipeline: parse -> embed -> store."""
    print("=" * 60)
    print("Step 1: Parsing PDFs")
    print("=" * 60)
    parse_pdfs()

    print("\n" + "=" * 60)
    print("Step 2: Building text embeddings and storing in ChromaDB")
    print("=" * 60)
    text_embedder = TextEmbedder()
    store = VectorStore()

    text_chunks_files = list(DATA_PROCESSED.rglob("text_chunks.json"))
    if not text_chunks_files:
        print("No text chunks found. Ensure PDFs are in data/raw/textbooks/")
        return

    all_chunks = []
    for chunk_file in text_chunks_files:
        with open(chunk_file) as f:
            chunks = json.load(f)
        source = chunk_file.parent.name
        for c in chunks:
            c["source"] = source
        all_chunks.extend(chunks)

    print(f"Embedding {len(all_chunks)} text chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = text_embedder.embed_batch(texts)
    ids = [c["chunk_id"] for c in all_chunks]
    metadatas = [{"page": c["page"], "source": c.get("source", "")} for c in all_chunks]

    store.add_texts(
        collection_name=TEXT_COLLECTION,
        texts=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )
    print(f"Stored {len(all_chunks)} text chunks in collection '{TEXT_COLLECTION}'")

    print("\n" + "=" * 60)
    print("Step 3: Building image embeddings and storing in ChromaDB")
    print("=" * 60)
    image_embedder = ImageEmbedder()

    image_meta_files = list(DATA_PROCESSED.rglob("images.json"))
    all_images = []
    for img_file in image_meta_files:
        with open(img_file) as f:
            imgs = json.load(f)
        source = img_file.parent.name
        for im in imgs:
            im["source"] = source
        all_images.extend(imgs)

    if not all_images:
        print("No images found to embed.")
        return

    to_embed: list[tuple[dict, Path]] = []
    for im in all_images:
        resolved = resolve_data_path(im["image_path"])
        if resolved is None or not resolved.is_file():
            print(f"Skipping missing image: {im.get('image_path')}")
            continue
        to_embed.append((im, resolved))

    print(f"Embedding {len(to_embed)} images...")
    image_paths = [str(r) for _, r in to_embed]
    img_embeddings = image_embedder.embed_images(image_paths)
    img_ids = [f"img_{i}" for i in range(len(to_embed))]
    img_metadatas = [
        {"page": im["page"], "source": im.get("source", ""), "image_path": path_for_storage(r)}
        for im, r in to_embed
    ]

    store.add_images(
        collection_name=IMAGE_COLLECTION,
        embeddings=img_embeddings,
        ids=img_ids,
        metadatas=img_metadatas,
    )
    print(f"Stored {len(all_images)} image embeddings in collection '{IMAGE_COLLECTION}'")

    print("\n" + "=" * 60)
    print("Knowledge base build complete!")
    print("=" * 60)


if __name__ == "__main__":
    build()
