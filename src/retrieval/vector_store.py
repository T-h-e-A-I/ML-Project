"""ChromaDB vector store for text and image embeddings."""

from pathlib import Path

import chromadb

from configs.default import CHROMA_PERSIST_DIR, TEXT_COLLECTION, IMAGE_COLLECTION


class VectorStore:
    """Manages ChromaDB collections for text and image embeddings."""

    def __init__(self, persist_dir: str | Path | None = None):
        persist_dir = str(persist_dir or CHROMA_PERSIST_DIR)
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)

    def _get_or_create(self, name: str, metadata: dict | None = None):
        return self.client.get_or_create_collection(
            name=name,
            metadata=metadata or {"hnsw:space": "cosine"},
        )

    def add_texts(
        self,
        collection_name: str,
        texts: list[str],
        embeddings: list[list[float]],
        ids: list[str],
        metadatas: list[dict] | None = None,
    ):
        """Add text documents with pre-computed embeddings."""
        col = self._get_or_create(collection_name)
        col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def add_images(
        self,
        collection_name: str,
        embeddings: list[list[float]],
        ids: list[str],
        metadatas: list[dict] | None = None,
    ):
        """Add image embeddings with metadata (paths stored in metadatas)."""
        col = self._get_or_create(collection_name)
        col.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query_text(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> dict:
        """Query text collection by embedding vector."""
        col = self._get_or_create(collection_name)
        return col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def query_image(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> dict:
        """Query image collection by embedding vector."""
        col = self._get_or_create(collection_name)
        return col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"],
        )

    def get_collection_count(self, collection_name: str) -> int:
        """Return the number of items in a collection."""
        try:
            col = self.client.get_collection(collection_name)
            return col.count()
        except Exception:
            return 0
