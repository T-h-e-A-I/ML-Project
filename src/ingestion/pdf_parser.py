"""Parse physics textbook PDFs: extract text chunks and embedded figures."""

import json
from pathlib import Path
from typing import Optional

import fitz  # pymupdf
from PIL import Image
from tqdm import tqdm

from configs.default import CHUNK_SIZE, CHUNK_OVERLAP, DATA_RAW, DATA_PROCESSED


def extract_text_and_images(
    pdf_path: str | Path,
    output_dir: Optional[str | Path] = None,
) -> dict:
    """
    Extract text and images from a PDF.
    Returns dict with 'text_chunks' and 'images' lists.
    """
    pdf_path = Path(pdf_path)
    if output_dir is None:
        output_dir = DATA_PROCESSED / pdf_path.stem
    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))

    all_text = []
    images = []

    for page_num in tqdm(range(len(doc)), desc=f"Parsing {pdf_path.name}"):
        page = doc[page_num]

        page_text = page.get_text("text")
        if page_text.strip():
            all_text.append({
                "page": page_num + 1,
                "text": page_text.strip(),
            })

        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                if base_image is None:
                    continue
                img_bytes = base_image["image"]
                img_ext = base_image.get("ext", "png")

                if len(img_bytes) < 5000:
                    continue

                img_filename = f"page{page_num + 1}_img{img_idx}.{img_ext}"
                img_path = img_dir / img_filename
                with open(img_path, "wb") as f:
                    f.write(img_bytes)

                images.append({
                    "page": page_num + 1,
                    "image_path": str(img_path),
                    "xref": xref,
                })
            except Exception:
                continue

    doc.close()

    text_chunks = _chunk_text(all_text)

    metadata = {
        "source_pdf": str(pdf_path),
        "num_pages": len(doc) if hasattr(doc, "__len__") else 0,
        "num_text_chunks": len(text_chunks),
        "num_images": len(images),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(output_dir / "text_chunks.json", "w") as f:
        json.dump(text_chunks, f, indent=2)
    with open(output_dir / "images.json", "w") as f:
        json.dump(images, f, indent=2)

    print(f"Extracted {len(text_chunks)} text chunks and {len(images)} images")
    return {"text_chunks": text_chunks, "images": images, "metadata": metadata}


def _chunk_text(
    page_texts: list[dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split page texts into overlapping chunks by approximate word count."""
    chunks = []
    chunk_id = 0

    for page_data in page_texts:
        words = page_data["text"].split()
        page_num = page_data["page"]

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])

            if len(chunk_text.strip()) > 20:
                chunks.append({
                    "chunk_id": f"chunk_{chunk_id}",
                    "page": page_num,
                    "text": chunk_text,
                    "start_word": start,
                    "end_word": end,
                })
                chunk_id += 1

            start += chunk_size - overlap

    return chunks


def build_knowledge_base(pdf_dir: Optional[str | Path] = None):
    """Process all PDFs in the raw data directory."""
    if pdf_dir is None:
        pdf_dir = DATA_RAW / "textbooks"

    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        print(f"No textbook directory found at {pdf_dir}. Skipping PDF parsing.")
        return []

    all_results = []
    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        print(f"\nProcessing: {pdf_file.name}")
        result = extract_text_and_images(pdf_file)
        all_results.append(result)

    return all_results


if __name__ == "__main__":
    build_knowledge_base()
