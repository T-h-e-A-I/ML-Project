"""Text-only RAG generator using Mistral-7B or API-based LLM."""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from configs.default import TEXT_LLM

RAG_PROMPT_TEMPLATE = """You are a physics expert. Use the provided context to answer the question accurately.
If the context is not sufficient, use your knowledge but clearly state when you are going beyond the provided context.
Provide step-by-step reasoning.

Context:
{context}

Question: {question}

Answer:"""


class TextGenerator:
    """Text-only generator for RAG baseline (B1 config)."""

    def __init__(self, model_name: str = TEXT_LLM, use_quantization: bool = True):
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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        question: str,
        context_chunks: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        """Generate an answer given a question and retrieved context chunks."""
        context_text = "\n\n".join(
            f"[Source: page {c.get('metadata', {}).get('page', '?')}]\n{c['text']}"
            for c in context_chunks
            if c.get("text")
        )

        prompt = RAG_PROMPT_TEMPLATE.format(context=context_text, question=question)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return generated.strip()


class APITextGenerator:
    """Fallback generator using OpenAI API (GPT-3.5-turbo)."""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate(
        self,
        question: str,
        context_chunks: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        context_text = "\n\n".join(
            f"[Source: page {c.get('metadata', {}).get('page', '?')}]\n{c['text']}"
            for c in context_chunks
            if c.get("text")
        )

        prompt = RAG_PROMPT_TEMPLATE.format(context=context_text, question=question)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
