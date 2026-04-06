"""Post-hoc analysis over saved prediction JSON files.

Usage:
  python -m src.evaluation.posthoc_analysis \
    --preds-dir data/preds \
    --output-dir data/eval/posthoc
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rouge_score import rouge_scorer
except ImportError:  # pragma: no cover
    rouge_scorer = None

try:
    from src.evaluation.metrics import compute_bert_score
except ImportError:  # pragma: no cover
    compute_bert_score = None


def _normalize(text: str) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r"[^\w\s:.-]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _first_sentence(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    parts = re.split(r"[.\n!?]+", t, maxsplit=1)
    return parts[0].strip()


def _token_f1(pred: str, ref: str) -> float:
    p = _normalize(pred).split()
    r = _normalize(ref).split()
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    p_count = Counter(p)
    r_count = Counter(r)
    common = sum((p_count & r_count).values())
    if common == 0:
        return 0.0
    prec = common / len(p)
    rec = common / len(r)
    return 2 * prec * rec / (prec + rec)


def _topic_tag(question: str) -> str:
    q = (question or "").lower()
    if "magnetic force" in q or "magnets" in q:
        return "magnetism"
    if "punnett" in q or "offspring" in q or "ratio" in q:
        return "genetics_ratio"
    if "kinetic energy" in q or "temperature" in q:
        return "thermal_kinetic"
    if "weather" in q or "climate" in q:
        return "weather_climate"
    if "which of the following could" in q:
        return "experimental_design"
    return "other"


def _compute_tfidf_cosines(preds: list[str], refs: list[str]) -> list[float]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        # Fallback lexical cosine without sklearn.
        values: list[float] = []
        for p, r in zip(preds, refs):
            p_tokens = Counter(_normalize(p).split())
            r_tokens = Counter(_normalize(r).split())
            all_keys = set(p_tokens) | set(r_tokens)
            dot = float(sum(p_tokens[k] * r_tokens[k] for k in all_keys))
            p_norm = float(np.sqrt(sum(v * v for v in p_tokens.values())))
            r_norm = float(np.sqrt(sum(v * v for v in r_tokens.values())))
            sim = dot / (p_norm * r_norm) if p_norm > 0 and r_norm > 0 else 0.0
            values.append(sim)
        return values

    values: list[float] = []
    for p, r in zip(preds, refs):
        docs = [p or ".", r or "."]
        vect = TfidfVectorizer(ngram_range=(1, 2))
        try:
            X = vect.fit_transform(docs)
            sim = float(cosine_similarity(X[0], X[1])[0][0])
        except ValueError:
            # Rare case: tokenizer drops all tokens (empty vocabulary)
            p_tokens = Counter(_normalize(p).split())
            r_tokens = Counter(_normalize(r).split())
            all_keys = set(p_tokens) | set(r_tokens)
            dot = float(sum(p_tokens[k] * r_tokens[k] for k in all_keys))
            p_norm = float(np.sqrt(sum(v * v for v in p_tokens.values())))
            r_norm = float(np.sqrt(sum(v * v for v in r_tokens.values())))
            sim = dot / (p_norm * r_norm) if p_norm > 0 and r_norm > 0 else 0.0
        values.append(sim)
    return values


def _lcs_len(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[n]


def _rouge_l_f1_simple(pred: str, ref: str) -> float:
    p_toks = _normalize(pred).split()
    r_toks = _normalize(ref).split()
    if not p_toks or not r_toks:
        return 0.0
    lcs = _lcs_len(p_toks, r_toks)
    if lcs == 0:
        return 0.0
    prec = lcs / len(p_toks)
    rec = lcs / len(r_toks)
    return 2 * prec * rec / (prec + rec)


def _analyze_rows(rows: list[dict], config_name: str) -> tuple[dict, pd.DataFrame]:
    questions = [str(r.get("question", "")) for r in rows]
    refs = [str(r.get("reference", "")) for r in rows]
    preds = [str(r.get("prediction", "")) for r in rows]
    preds_first = [_first_sentence(p) for p in preds]

    if rouge_scorer is not None:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_full = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(preds, refs)]
        rouge_first = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(preds_first, refs)]
    else:
        rouge_full = [_rouge_l_f1_simple(pred, ref) for pred, ref in zip(preds, refs)]
        rouge_first = [_rouge_l_f1_simple(pred, ref) for pred, ref in zip(preds_first, refs)]

    token_f1_full = [_token_f1(p, r) for p, r in zip(preds, refs)]
    token_f1_first = [_token_f1(p, r) for p, r in zip(preds_first, refs)]

    exact_norm = [float(_normalize(p) == _normalize(r)) for p, r in zip(preds, refs)]
    contains_ref = [float(_normalize(r) in _normalize(p)) for p, r in zip(preds, refs)]
    contains_ref_first = [float(_normalize(r) in _normalize(p)) for p, r in zip(preds_first, refs)]

    lengths = [len((p or "").split()) for p in preds]
    repetition_flag = [
        float(bool(re.search(r"(.{25,}?)(\1){2,}", (p or "").replace("\n", " "), flags=re.I)))
        for p in preds
    ]
    therefore_flag = [float("therefore, the answer is" in (p or "").lower()) for p in preds]
    visual_flag = [
        float(any(k in (p or "").lower() for k in ["visual elements", "visual aid", "diagram"]))
        for p in preds
    ]
    empty_flag = [float(not (p or "").strip()) for p in preds]
    tfidf_cos = _compute_tfidf_cosines(preds, refs)

    bert_f1 = float("nan")
    if compute_bert_score is not None:
        try:
            bs = compute_bert_score(preds, refs)
            bert_f1 = float(bs.get("f1", 0.0))
        except Exception:
            bert_f1 = float("nan")

    topics = [_topic_tag(q) for q in questions]

    summary = {
        "config": config_name,
        "num_samples": len(rows),
        "exact_norm_acc": float(np.mean(exact_norm)),
        "contains_ref_acc": float(np.mean(contains_ref)),
        "contains_ref_first_sentence_acc": float(np.mean(contains_ref_first)),
        "token_f1_mean": float(np.mean(token_f1_full)),
        "token_f1_first_sentence_mean": float(np.mean(token_f1_first)),
        "rougeL_f1_mean": float(np.mean(rouge_full)),
        "rougeL_f1_first_sentence_mean": float(np.mean(rouge_first)),
        "bert_score_f1_mean": bert_f1,
        "tfidf_cosine_mean": float(np.mean(tfidf_cos)),
        "avg_pred_words": float(np.mean(lengths)),
        "median_pred_words": float(np.median(lengths)),
        "p95_pred_words": float(np.percentile(lengths, 95)),
        "empty_rate": float(np.mean(empty_flag)),
        "repetition_rate": float(np.mean(repetition_flag)),
        "therefore_phrase_rate": float(np.mean(therefore_flag)),
        "visual_phrase_rate": float(np.mean(visual_flag)),
    }

    detail_df = pd.DataFrame(
        {
            "config": config_name,
            "question": questions,
            "reference": refs,
            "prediction": preds,
            "prediction_first_sentence": preds_first,
            "topic": topics,
            "pred_words": lengths,
            "exact_norm": exact_norm,
            "contains_ref": contains_ref,
            "contains_ref_first_sentence": contains_ref_first,
            "token_f1": token_f1_full,
            "token_f1_first_sentence": token_f1_first,
            "rougeL_f1": rouge_full,
            "rougeL_f1_first_sentence": rouge_first,
            "tfidf_cosine": tfidf_cos,
            "is_empty": empty_flag,
            "has_repetition": repetition_flag,
            "has_therefore_phrase": therefore_flag,
            "has_visual_phrase": visual_flag,
        }
    )
    return summary, detail_df


def _parse_config_name(path: Path) -> str:
    # e.g. preds_M2_0_200.json -> M2
    m = re.search(r"preds_([A-Za-z0-9]+)", path.stem)
    return m.group(1) if m else path.stem


def run(preds_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_files = sorted(preds_dir.glob("preds_*.json"))
    if not pred_files:
        raise FileNotFoundError(f"No preds_*.json found in {preds_dir}")

    all_summaries: list[dict] = []
    all_details: list[pd.DataFrame] = []

    for p in pred_files:
        with open(p, encoding="utf-8") as f:
            rows = json.load(f)
        cfg = _parse_config_name(p)
        summary, detail_df = _analyze_rows(rows, cfg)
        summary["source_file"] = str(p)
        all_summaries.append(summary)
        all_details.append(detail_df)

        detail_path = output_dir / f"question_metrics_{cfg}.csv"
        detail_df.to_csv(detail_path, index=False)
        print(f"Saved question metrics: {detail_path}")

    summary_df = pd.DataFrame(all_summaries).sort_values("config")
    summary_csv = output_dir / "posthoc_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary CSV: {summary_csv}")

    with open(output_dir / "posthoc_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Saved summary JSON: {output_dir / 'posthoc_summary.json'}")

    merged = pd.concat(all_details, ignore_index=True)
    topic_agg = (
        merged.groupby(["config", "topic"], as_index=False)
        .agg(
            n=("question", "count"),
            contains_ref=("contains_ref", "mean"),
            token_f1=("token_f1", "mean"),
            rougeL_f1=("rougeL_f1", "mean"),
            pred_words=("pred_words", "mean"),
            repetition=("has_repetition", "mean"),
        )
        .sort_values(["config", "n"], ascending=[True, False])
    )
    topic_csv = output_dir / "topic_breakdown.csv"
    topic_agg.to_csv(topic_csv, index=False)
    print(f"Saved topic breakdown: {topic_csv}")


def main():
    parser = argparse.ArgumentParser(description="Post-hoc metrics over saved preds JSON files.")
    parser.add_argument("--preds-dir", type=Path, default=Path("data/preds"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval/posthoc"))
    args = parser.parse_args()
    run(args.preds_dir, args.output_dir)


if __name__ == "__main__":
    main()
