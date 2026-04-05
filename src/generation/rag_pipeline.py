"""End-to-end RAG pipeline orchestrating retrieval and generation."""

from pathlib import Path

from configs.default import FUSION_ALPHA, TOP_K, resolve_data_path
from src.retrieval.multimodal_retriever import MultimodalRetriever
from src.generation.text_generator import TextGenerator
from src.generation.vlm_generator import VLMGenerator


class RAGPipeline:
    """
    Unified RAG pipeline supporting all experimental configurations.

    Configs:
        B1: text retrieval + text LLM
        B2: text retrieval + VLM
        M1: multimodal retrieval + VLM
        M2: multimodal retrieval + fine-tuned VLM
        M3: no retrieval + fine-tuned VLM
    """

    def __init__(
        self,
        config: str = "M1",
        adapter_path: str | None = None,
        top_k: int = TOP_K,
        alpha: float = FUSION_ALPHA,
    ):
        self.config = config
        self.top_k = top_k
        self.alpha = alpha

        use_retrieval = config not in ("M3",)
        use_vlm = config not in ("B1",)
        use_finetuned = config in ("M2", "M3")

        self.retriever = MultimodalRetriever() if use_retrieval else None

        if use_vlm:
            self.generator = VLMGenerator(
                adapter_path=adapter_path if use_finetuned else None
            )
        else:
            self.generator = TextGenerator()

    def query(
        self,
        question: str,
        image_path: str | Path | None = None,
    ) -> dict:
        """
        Run the full RAG pipeline for a single query.

        Returns dict with 'answer', 'retrieved_context', and 'retrieved_images'.
        """
        context_chunks = []
        image_paths = []
        query_img = resolve_data_path(image_path) if image_path else None

        if self.retriever:
            if self.config == "B1" or self.config == "B2":
                context_chunks = self.retriever.retrieve_text_only(
                    question, top_k=self.top_k
                )
            else:
                results = self.retriever.retrieve_multimodal(
                    question,
                    query_image_path=str(query_img) if query_img else None,
                    top_k=self.top_k,
                    alpha=self.alpha,
                )
                context_chunks = results["text_results"]
                image_paths = []
                for r in results["image_results"]:
                    ip = r.get("image_path")
                    rp = resolve_data_path(ip) if ip else None
                    if rp and rp.is_file():
                        image_paths.append(str(rp))

        if query_img and query_img.is_file():
            image_paths = [str(query_img)] + image_paths[:2]

        if isinstance(self.generator, TextGenerator):
            answer = self.generator.generate(question, context_chunks)
        else:
            answer = self.generator.generate(
                question,
                context_chunks=context_chunks,
                image_paths=image_paths or None,
            )

        return {
            "answer": answer,
            "retrieved_context": context_chunks,
            "retrieved_images": image_paths,
            "config": self.config,
        }
