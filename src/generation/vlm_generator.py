"""Vision-Language Model generator using LLaVA for multimodal RAG."""

from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
)

from configs.default import VLM_MODEL, resolve_data_path

MULTIMODAL_RAG_PROMPT = """You are a physics expert. Use the provided context and any diagrams to answer the question.
Provide step-by-step reasoning, referencing visual elements when relevant.

Text Context:
{context}

Question: {question}

Answer:"""


class VLMGenerator:
    """LLaVA-based multimodal generator for RAG (B2, M1, M2 configs)."""

    def __init__(
        self,
        model_name: str = VLM_MODEL,
        use_quantization: bool = True,
        adapter_path: str | None = None,
    ):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        quantization_config = None
        if use_quantization and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            print(f"Loaded LoRA adapter from {adapter_path}")

    def generate(
        self,
        question: str,
        context_chunks: list[dict] | None = None,
        image_paths: list[str | Path] | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        """
        Generate answer with optional text context and images.

        Args:
            question: The physics question.
            context_chunks: Retrieved text chunks (list of dicts with 'text' key).
            image_paths: Paths to relevant images/diagrams.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
        """
        context_text = ""
        if context_chunks:
            context_text = "\n\n".join(
                f"[Source: page {c.get('metadata', {}).get('page', '?')}]\n{c['text']}"
                for c in context_chunks
                if c.get("text")
            )

        prompt = MULTIMODAL_RAG_PROMPT.format(
            context=context_text or "(No text context provided)",
            question=question,
        )

        images = []
        if image_paths:
            for p in image_paths[:3]:  # LLaVA 1.5 handles single image best
                try:
                    rp = resolve_data_path(p)
                    if rp is None or not rp.is_file():
                        raise FileNotFoundError(p)
                    img = Image.open(rp).convert("RGB")
                    images.append(img)
                except Exception as e:
                    print(f"Could not load image {p}: {e}")

        if images:
            image_tokens = "<image>\n" * len(images)
            prompt = f"USER: {image_tokens}{prompt}\nASSISTANT:"
            inputs = self.processor(
                text=prompt,
                images=images,
                return_tensors="pt",
                padding=True,
            )
        else:
            prompt = f"USER: {prompt}\nASSISTANT:"
            inputs = self.processor(
                text=prompt,
                return_tensors="pt",
                padding=True,
            )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        return generated.strip()
