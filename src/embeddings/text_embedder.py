"""Text embedding using SentenceTransformers."""

from sentence_transformers import SentenceTransformer

from configs.default import TEXT_EMBED_MODEL


class TextEmbedder:
    """Wraps a SentenceTransformer model for text embedding."""

    def __init__(self, model_name: str = TEXT_EMBED_MODEL):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Embed a batch of text strings."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.tolist()
