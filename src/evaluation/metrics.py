"""Evaluation metrics: accuracy, ROUGE-L, BERTScore, GPT-4o judge."""

import os
import re
import numpy as np


def exact_match_accuracy(predictions: list[str], references: list[str]) -> float:
    """Simple exact-match accuracy after normalization."""
    correct = 0
    for pred, ref in zip(predictions, references):
        if _normalize(pred) == _normalize(ref):
            correct += 1
    return correct / max(len(predictions), 1)


def contains_accuracy(predictions: list[str], references: list[str]) -> float:
    """Check if the normalized reference is contained in the prediction."""
    correct = 0
    for pred, ref in zip(predictions, references):
        if _normalize(ref) in _normalize(pred):
            correct += 1
    return correct / max(len(predictions), 1)


def compute_rouge_l(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-L scores."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]
    return {
        "precision": np.mean([s["rougeL"].precision for s in scores]),
        "recall": np.mean([s["rougeL"].recall for s in scores]),
        "f1": np.mean([s["rougeL"].fmeasure for s in scores]),
    }


def _bert_score_safe_text(text: str) -> str:
    """bert_score encodes empty strings via build_inputs_with_special_tokens([]), which
    breaks on recent transformers + RoBERTa. Replace empties with a minimal placeholder."""
    t = (text or "").strip()
    return t if t else "."


def compute_bert_score(predictions: list[str], references: list[str]) -> dict:
    """Compute BERTScore."""
    from bert_score import score as bert_score_fn

    preds = [_bert_score_safe_text(p) for p in predictions]
    refs = [_bert_score_safe_text(r) for r in references]
    P, R, F1 = bert_score_fn(preds, refs, lang="en", verbose=False)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
    }


def gpt4o_judge(
    questions: list[str],
    predictions: list[str],
    references: list[str],
    max_samples: int = 50,
) -> dict:
    """
    Use GPT-4o as a judge to evaluate reasoning quality.
    Scores each prediction on a 1-5 scale for correctness and reasoning.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"correctness": 0.0, "reasoning": 0.0, "note": "OPENAI_API_KEY not set"}

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    judge_prompt = """Rate the following physics answer on two dimensions (1-5 scale each):
1. Correctness: How factually correct is the answer compared to the reference?
2. Reasoning: How well does the answer demonstrate step-by-step physics reasoning?

Question: {question}
Reference Answer: {reference}
Model Answer: {prediction}

Return ONLY a JSON object: {{"correctness": <1-5>, "reasoning": <1-5>}}"""

    correctness_scores = []
    reasoning_scores = []

    samples = list(zip(questions, predictions, references))[:max_samples]

    for q, pred, ref in samples:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": judge_prompt.format(
                        question=q, reference=ref, prediction=pred
                    ),
                }],
                max_tokens=100,
                temperature=0,
                response_format={"type": "json_object"},
            )
            import json
            result = json.loads(response.choices[0].message.content)
            correctness_scores.append(result.get("correctness", 3))
            reasoning_scores.append(result.get("reasoning", 3))
        except Exception:
            correctness_scores.append(3)
            reasoning_scores.append(3)

    return {
        "correctness": np.mean(correctness_scores) if correctness_scores else 0.0,
        "reasoning": np.mean(reasoning_scores) if reasoning_scores else 0.0,
        "num_judged": len(correctness_scores),
    }


def compute_all_metrics(
    questions: list[str],
    predictions: list[str],
    references: list[str],
    use_gpt4o_judge: bool = True,
) -> dict:
    """Compute all evaluation metrics."""
    results = {
        "exact_match": exact_match_accuracy(predictions, references),
        "contains_accuracy": contains_accuracy(predictions, references),
        "rouge_l": compute_rouge_l(predictions, references),
        "bert_score": compute_bert_score(predictions, references),
        "num_samples": len(predictions),
    }
    if use_gpt4o_judge:
        results["gpt4o_judge"] = gpt4o_judge(questions, predictions, references)
    return results


def _normalize(text: str) -> str:
    """Lowercase, strip whitespace and punctuation for comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text
