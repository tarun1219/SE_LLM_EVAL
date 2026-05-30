#!/usr/bin/env python3
"""
run_metrics_pure.py — Zero-ML metrics computation for Qasper responses.

Uses only pure-Python libraries — no sentence-transformers, no GPT-2, no torch.
Safe on memory-constrained machines.

BERTScore substitute: TF-IDF cosine similarity (noted in paper as a limitation).
Fluency substitute:   type-token ratio (inverse = higher tokens reused = less fluent).
Relevance:            TF-IDF cosine similarity between question and response.

Outputs:
    results/qasper_automatic_metrics.csv
    results/qasper_stylistic_metrics.csv
"""
from __future__ import annotations
import time
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from rouge import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download needed NLTK data silently
for pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
    nltk.download(pkg, quiet=True)

REPO     = Path(__file__).parent
RESP_DIR = REPO / "Qasper_analysis" / "responses"
OUT_DIR  = REPO / "results"
OUT_DIR.mkdir(exist_ok=True)

_rouge = Rouge()


# ─── Automatic metrics ──────────────────────────────────────────────────────

def bleu_scores(references: list[str], candidate: str) -> dict:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    sf = SmoothingFunction().method1
    ref_tok = [nltk.word_tokenize(r.lower()) for r in references if r.strip()]
    can_tok = nltk.word_tokenize(candidate.lower())
    if not ref_tok or not can_tok:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}
    return {
        "bleu_1": sentence_bleu(ref_tok, can_tok, weights=(1,0,0,0), smoothing_function=sf),
        "bleu_2": sentence_bleu(ref_tok, can_tok, weights=(.5,.5,0,0), smoothing_function=sf),
        "bleu_3": sentence_bleu(ref_tok, can_tok, weights=(.33,.33,.33,0), smoothing_function=sf),
        "bleu_4": sentence_bleu(ref_tok, can_tok, weights=(.25,.25,.25,.25), smoothing_function=sf),
    }


def rouge_scores(references: list[str], candidate: str) -> dict:
    best = {"rouge_1_f": 0.0, "rouge_2_f": 0.0, "rouge_l_f": 0.0}
    cand = candidate.strip() or "."
    for ref in references:
        ref = ref.strip() or "."
        try:
            s = _rouge.get_scores(cand, ref)[0]
            best["rouge_1_f"] = max(best["rouge_1_f"], s["rouge-1"]["f"])
            best["rouge_2_f"] = max(best["rouge_2_f"], s["rouge-2"]["f"])
            best["rouge_l_f"] = max(best["rouge_l_f"], s["rouge-l"]["f"])
        except Exception:
            pass
    return best


def meteor_score(references: list[str], candidate: str) -> float:
    from nltk.translate.meteor_score import meteor_score as ms
    ref_tok = [nltk.word_tokenize(r.lower()) for r in references if r.strip()]
    can_tok = nltk.word_tokenize(candidate.lower())
    if not ref_tok or not can_tok:
        return 0.0
    return float(ms(ref_tok, can_tok))


def tfidf_cosine(texts_a: list[str], texts_b: list[str]) -> list[float]:
    """Pairwise TF-IDF cosine similarity. Safe fallback when embeddings unavailable."""
    scores = []
    for a, b in zip(texts_a, texts_b):
        try:
            vec = TfidfVectorizer().fit_transform([a or ".", b or "."])
            sim = float(cosine_similarity(vec[0:1], vec[1:2])[0][0])
        except Exception:
            sim = 0.0
        scores.append(sim)
    return scores


# ─── Stylistic metrics ────────────────────────────────────────────────────────

def formality_mtld(text: str) -> float:
    from lexical_diversity import lex_div as ld
    try:
        tokens = ld.flemmatize(text)
        return float(ld.mtld(tokens)) if tokens else 0.0
    except Exception:
        return 0.0


def readability_flesch(text: str) -> float:
    import textstat
    try:
        return float(textstat.flesch_reading_ease(text))
    except Exception:
        return 0.0


def correctness_fuzzy(reference: str, candidate: str) -> float:
    from fuzzywuzzy import fuzz
    try:
        return fuzz.token_set_ratio(reference or ".", candidate or ".") / 100.0
    except Exception:
        return 0.0


def fluency_ttr(text: str) -> float:
    """Type-token ratio as fluency proxy. Higher TTR = more vocabulary variety."""
    tokens = nltk.word_tokenize(text.lower())
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


# ─── Process one CSV ─────────────────────────────────────────────────────────

def process_file(fp: Path) -> tuple[dict | None, dict | None]:
    df = pd.read_csv(fp)
    if df.empty:
        return None, None

    candidates  = df["response"].fillna("").tolist()
    gold_col    = [g.split(" ||| ") for g in df["gold_answer"].fillna("").tolist()]
    single_refs = [r[0] if r else "" for r in gold_col]
    questions   = df["question"].fillna("").tolist()

    model  = df["model"].iloc[0]         if "model"     in df.columns else fp.stem
    prompt = int(df["prompt_num"].iloc[0]) if "prompt_num" in df.columns else 0
    shot   = df["shot_type"].iloc[0]     if "shot_type"  in df.columns else ""
    year   = int(df["year"].iloc[0])     if "year"       in df.columns else 2026
    n      = len(df)

    # ── Automatic metrics ─────────────────────────────────────────────
    bleus   = [bleu_scores(g, c) for g, c in zip(gold_col, candidates)]
    rouges  = [rouge_scores(g, c) for g, c in zip(gold_col, candidates)]
    meteors = [meteor_score(g, c) for g, c in zip(gold_col, candidates)]

    # TF-IDF cosine as BERTScore proxy (same per-pair, average over file)
    tfidf_sim = tfidf_cosine(candidates, single_refs)

    auto = {
        "model": model, "prompt_num": prompt, "shot_type": shot, "year": year,
        "bleu_1":    float(np.mean([b["bleu_1"]    for b in bleus])),
        "bleu_2":    float(np.mean([b["bleu_2"]    for b in bleus])),
        "bleu_3":    float(np.mean([b["bleu_3"]    for b in bleus])),
        "bleu_4":    float(np.mean([b["bleu_4"]    for b in bleus])),
        "rouge_1_f": float(np.mean([r["rouge_1_f"] for r in rouges])),
        "rouge_2_f": float(np.mean([r["rouge_2_f"] for r in rouges])),
        "rouge_l_f": float(np.mean([r["rouge_l_f"] for r in rouges])),
        "meteor":    float(np.mean(meteors)),
        "bertscore_p":   float(np.mean(tfidf_sim)),   # TF-IDF proxy
        "bertscore_r":   float(np.mean(tfidf_sim)),
        "bertscore_f1":  float(np.mean(tfidf_sim)),
        "n": n,
        "source": "2026",
        "note": "bertscore=tfidf_cosine_proxy",
    }

    # ── Stylistic metrics ─────────────────────────────────────────────
    qa_rel  = tfidf_cosine(questions, candidates)    # question–response relevance
    ca_rel  = tfidf_cosine(single_refs, candidates)  # gold–response relevance
    flu     = [fluency_ttr(c) for c in candidates]   # type-token ratio
    form    = [formality_mtld(c) for c in candidates]
    read    = [readability_flesch(c) for c in candidates]
    corr    = [correctness_fuzzy(g, c) for g, c in zip(single_refs, candidates)]

    sty = {
        "model": model, "prompt_num": prompt, "shot_type": shot, "year": year,
        "qa_relevance":       float(np.mean(qa_rel)),
        "ca_relevance":       float(np.mean(ca_rel)),
        "fluency_ppl":        float(np.mean(flu)),    # actually TTR, column name kept for compat
        "formality_mtld":     float(np.mean(form)),
        "readability_flesch": float(np.mean(read)),
        "correctness_ner":    float(np.mean(corr)),   # fuzzywuzzy, column name kept for compat
        "judge_correctness":  float("nan"),
        "judge_relevance":    float("nan"),
        "judge_human_likeness": float("nan"),
        "n": n,
        "source": "2026",
    }

    return auto, sty


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    files = sorted(RESP_DIR.glob("*.csv"))
    files = [f for f in files if "metadata" not in f.name.lower()]
    print(f"Found {len(files)} Qasper response CSVs.", flush=True)
    print("Using pure-Python metrics (no ML models).", flush=True)

    auto_rows, sty_rows = [], []
    for i, fp in enumerate(files):
        t0 = time.time()
        print(f"  [{i+1}/{len(files)}] {fp.name} ...", end=" ", flush=True)
        try:
            auto, sty = process_file(fp)
            if auto:
                auto_rows.append(auto)
            if sty:
                sty_rows.append(sty)
            print(f"done ({time.time()-t0:.1f}s)", flush=True)
        except Exception as exc:
            import traceback
            print(f"ERROR: {exc}", flush=True)
            traceback.print_exc()

    if auto_rows:
        out = OUT_DIR / "qasper_automatic_metrics.csv"
        pd.DataFrame(auto_rows).to_csv(out, index=False)
        print(f"\n✓ Saved -> {out}", flush=True)
    if sty_rows:
        out = OUT_DIR / "qasper_stylistic_metrics.csv"
        pd.DataFrame(sty_rows).to_csv(out, index=False)
        print(f"✓ Saved -> {out}", flush=True)

    print(f"\nProcessed {len(auto_rows)}/{len(files)} files successfully.", flush=True)


if __name__ == "__main__":
    main()
