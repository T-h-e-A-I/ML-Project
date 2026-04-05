"""Multimodal retriever: fuses text and image retrieval results."""

from pathlib import Path

from src.embeddings.text_embedder import TextEmbedder
from src.embeddings.image_embedder import ImageEmbedder
from src.retrieval.vector_store import VectorStore
from configs.default import (
    FUSION_ALPHA,
    IMAGE_COLLECTION,
    TEXT_COLLECTION,
    TOP_K,
    resolve_data_path,
)


class MultimodalRetriever:
    """
    Retrieves relevant text chunks and images using late fusion.

    For text-only queries: retrieve by text similarity.
    For queries with diagrams: fuse text and image similarities.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        text_embedder: TextEmbedder | None = None,
        image_embedder: ImageEmbedder | None = None,
    ):
        self.store = vector_store or VectorStore()
        self.text_embedder = text_embedder or TextEmbedder()
        self.image_embedder = image_embedder or ImageEmbedder()

    def retrieve_text_only(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> list[dict]:
        """Retrieve text chunks using text similarity only."""
        query_emb = self.text_embedder.embed(query)
        results = self.store.query_text(TEXT_COLLECTION, query_emb, top_k=top_k)
        return self._format_text_results(results)

    def retrieve_multimodal(
        self,
        query: str,
        query_image_path: str | Path | None = None,
        top_k: int = TOP_K,
        alpha: float = FUSION_ALPHA,
    ) -> dict:
        """
        Retrieve using both text and image similarity.

        Returns dict with 'text_results' and 'image_results'.
        alpha controls fusion: score = alpha * text_sim + (1-alpha) * image_sim
        """
        text_query_emb = self.text_embedder.embed(query)

        text_results = self.store.query_text(
            TEXT_COLLECTION, text_query_emb, top_k=top_k * 2
        )

        image_results = {"ids": [[]], "metadatas": [[]], "distances": [[]]}
        qimg = resolve_data_path(query_image_path) if query_image_path else None
        if qimg is not None and qimg.is_file():
            img_query_emb = self.image_embedder.embed_image(qimg)
            image_results = self.store.query_image(
                IMAGE_COLLECTION, img_query_emb, top_k=top_k
            )
        else:
            clip_text_emb = self.image_embedder.embed_text(query)
            image_results = self.store.query_image(
                IMAGE_COLLECTION, clip_text_emb, top_k=top_k
            )

        formatted_text = self._format_text_results(text_results)
        formatted_images = self._format_image_results(image_results)

        if qimg and formatted_text and formatted_images:
            formatted_text = self._rerank_with_fusion(
                formatted_text, formatted_images, alpha, top_k
            )

        return {
            "text_results": formatted_text[:top_k],
            "image_results": formatted_images[:top_k],
        }

    def _format_text_results(self, results: dict) -> list[dict]:
        """Format ChromaDB text query results."""
        formatted = []
        if not results["ids"] or not results["ids"][0]:
            return formatted
        for i, doc_id in enumerate(results["ids"][0]):
            formatted.append({
                "id": doc_id,
                "text": results["documents"][0][i] if results.get("documents") else "",
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                "distance": results["distances"][0][i] if results.get("distances") else 1.0,
                "score": 1.0 - (results["distances"][0][i] if results.get("distances") else 1.0),
            })
        return sorted(formatted, key=lambda x: x["score"], reverse=True)

    def _format_image_results(self, results: dict) -> list[dict]:
        """Format ChromaDB image query results."""
        formatted = []
        if not results["ids"] or not results["ids"][0]:
            return formatted
        for i, img_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            formatted.append({
                "id": img_id,
                "image_path": meta.get("image_path", ""),
                "metadata": meta,
                "distance": results["distances"][0][i] if results.get("distances") else 1.0,
                "score": 1.0 - (results["distances"][0][i] if results.get("distances") else 1.0),
            })
        return sorted(formatted, key=lambda x: x["score"], reverse=True)

    @staticmethod
    def _rerank_with_fusion(
        text_results: list[dict],
        image_results: list[dict],
        alpha: float,
        top_k: int,
    ) -> list[dict]:
        """Re-rank text results using late fusion with image scores."""
        image_scores = {}
        for img in image_results:
            page = img["metadata"].get("page")
            if page is not None:
                image_scores[page] = max(image_scores.get(page, 0), img["score"])

        for tr in text_results:
            page = tr["metadata"].get("page")
            img_score = image_scores.get(page, 0)
            tr["fused_score"] = alpha * tr["score"] + (1 - alpha) * img_score

        return sorted(text_results, key=lambda x: x.get("fused_score", x["score"]), reverse=True)[:top_k]
