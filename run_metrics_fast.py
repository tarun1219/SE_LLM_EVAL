#!/usr/bin/env python3
"""
run_metrics_fast.py — Memory-efficient Qasper metrics computation.

Uses only cached models (sentence-transformers all-mpnet-base-v2, GPT-2).
BERTScore replaced with sentence-transformer cosine F1 to avoid downloading
1.5GB DeBERTa on a memory-constrained machine.

Outputs:
    results/qasper_automatic_metrics.csv
    results/qasper_stylistic_metrics.csv
"""
from __future__ import annotations
import csv
import sys
import time
from pathlib import Path

import nltk
import numpy as np
import pandas as pd

for pkg in ("punkt", "wordnet", "omw-1.4", "punkt_tab"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

REPO     = Path(__file__).parent
RESP_DIR = REPO / "Qasper_analysis" / "responses"
OUT_DIR  = REPO / "results"
OUT_DIR.mkdir(exist_ok=True)


# ─── Automatic metrics ──────────────────────────────────────────────────────

def compute_bleu(references, candidate):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    sf = SmoothingFunction().method1
    ref_tokens = [nltk.word_tokenize(r.lower()) for r in references]
    cand_tokens = nltk.word_tokenize(candidate.lower())
    return {
        "bleu_1": sentence_bleu(ref_tokens, cand_tokens, weights=(1,0,0,0), smoothing_function=sf),
        "bleu_2": sentence_bleu(ref_tokens, cand_tokens, weights=(.5,.5,0,0), smoothing_function=sf),
        "bleu_3": sentence_bleu(ref_tokens, cand_tokens, weights=(.33,.33,.33,0), smoothing_function=sf),
        "bleu_4": sentence_bleu(ref_tokens, cand_tokens, weights=(.25,.25,.25,.25), smoothing_function=sf),
    }


def compute_rouge(references, candidate):
    from rouge import Rouge
    rouge = Rouge()
    best = {"rouge_1_f": 0.0, "rouge_2_f": 0.0, "rouge_l_f": 0.0}
    for ref in references:
        try:
            s = rouge.get_scores(candidate or ".", ref or ".")[0]
            best["rouge_1_f"] = max(best["rouge_1_f"], s["rouge-1"]["f"])
            best["rouge_2_f"] = max(best["rouge_2_f"], s["rouge-2"]["f"])
            best["rouge_l_f"] = max(best["rouge_l_f"], s["rouge-l"]["f"])
        except Exception:
            pass
    return best


def compute_meteor(references, candidate):
    from nltk.translate.meteor_score import meteor_score
    ref_tokens = [nltk.word_tokenize(r.lower()) for r in references]
    cand_tokens = nltk.word_tokenize(candidate.lower())
    return float(meteor_score(ref_tokens, cand_tokens))


# ─── Sentence-transformer proxy (replaces BERTScore + relevance) ─────────────

class STModel:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            from sentence_transformers import SentenceTransformer
            print("  Loading all-mpnet-base-v2 (cached)…", flush=True)
            cls._instance = SentenceTransformer("all-mpnet-base-v2")
        return cls._instance

    @classmethod
    def cosine(cls, a: str, b: str) -> float:
        from sentence_transformers import util
        m = cls.get()
        ea = m.encode(a, convert_to_tensor=True)
        eb = m.encode(b, convert_to_tensor=True)
        return float(util.pytorch_cos_sim(ea, eb).item())

    @classmethod
    def batch_cosine_f1(cls, candidates, references):
        """Compute P/R/F1 like BERTScore but using sentence-transformer embeddings."""
        m = cls.get()
        cand_embs = m.encode(candidates, convert_to_tensor=True, show_progress_bar=False)
        ref_embs  = m.encode(references,  convert_to_tensor=True, show_progress_bar=False)
        from sentence_transformers import util
        cos_mat = util.pytorch_cos_sim(cand_embs, ref_embs).diag()  # pairwise
        cos_vals = [max(float(v), 0.0) for v in cos_mat.tolist()]
        P = float(np.mean(cos_vals))
        R = float(np.mean(cos_vals))   # symmetric cosine → P==R per pair
        F1 = float(np.mean(cos_vals))
        return {"bertscore_p": P, "bertscore_r": R, "bertscore_f1": F1}


# ─── GPT-2 fluency ───────────────────────────────────────────────────────────

class GPT2Fluency:
    _instance = None

    @classmethod
    def load(cls):
        if cls._instance is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("  Loading GPT-2 (cached)…", flush=True)
            tok   = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            model.eval()
            cls._instance = (tok, model)
        return cls._instance

    @classmethod
    def ppl(cls, text: str) -> float:
        import torch
        if not text.strip():
            return 1000.0
        tok, model = cls.load()
        inputs = tok(text[:512], return_tensors="pt", truncation=True)
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss
        return float(torch.exp(loss).item())


# ─── Stylistic helpers ────────────────────────────────────────────────────────

def formality(text: str) -> float:
    from lexical_diversity import lex_div as ld
    try:
        tokens = ld.flemmatize(text)
        return float(ld.mtld(tokens)) if tokens else 0.0
    except Exception:
        return 0.0


def readability(text: str) -> float:
    import textstat
    return float(textstat.flesch_reading_ease(text))


def correctness_fuzzy(reference: str, candidate: str) -> float:
    from fuzzywuzzy import fuzz
    return fuzz.token_set_ratio(reference, candidate) / 100.0


# ─── Process one CSV ─────────────────────────────────────────────────────────

def process_file(fp: Path):
    df = pd.read_csv(fp)
    if df.empty:
        return None, None

    candidates  = df["response"].fillna("").tolist()
    gold_col    = [g.split(" ||| ") for g in df["gold_answer"].fillna("").tolist()]
    single_refs = [r[0] if r else "" for r in gold_col]
    questions   = df["question"].fillna("").tolist()

    model    = df["model"].iloc[0]    if "model"     in df.columns else fp.stem
    prompt   = int(df["prompt_num"].iloc[0]) if "prompt_num" in df.columns else 0
    shot     = df["shot_type"].iloc[0]       if "shot_type"  in df.columns else ""
    year     = int(df["year"].iloc[0])       if "year"       in df.columns else 2026
    n        = len(df)

    # ── Automatic ───────────────────────────────────────────────────
    bleus   = [compute_bleu(g, c) for g, c in zip(gold_col, candidates)]
    rouges  = [compute_rouge(g, c) for g, c in zip(gold_col, candidates)]
    meteors = [compute_meteor(g, c) for g, c in zip(gold_col, candidates)]

    # Sentence-transformer cosine similarity as BERTScore proxy
    bs = STModel.batch_cosine_f1(candidates, single_refs)

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
        **bs,
        "n": n,
        "note": "bertscore_proxy=sentence-transformer-cosine",
    }

    # ── Stylistic ─────────────────────────────────────────────────────
    qa_rel  = [STModel.cosine(q, c) for q, c in zip(questions,  candidates)]
    ca_rel  = [STModel.cosine(g, c) for g, c in zip(single_refs, candidates)]
    flu     = [GPT2Fluency.ppl(c)   for c in candidates]
    form    = [formality(c)          for c in candidates]
    read    = [readability(c)        for c in candidates]
    corr    = [correctness_fuzzy(g, c) for g, c in zip(single_refs, candidates)]

    sty = {
        "model": model, "prompt_num": prompt, "shot_type": shot, "year": year,
        "qa_relevance":       float(np.mean(qa_rel)),
        "ca_relevance":       float(np.mean(ca_rel)),
        "fluency_ppl":        float(np.mean(flu)),
        "formality_mtld":     float(np.mean(form)),
        "readability_flesch": float(np.mean(read)),
        "correctness_ner":    float(np.mean(corr)),
        "judge_correctness":  float("nan"),
        "judge_relevance":    float("nan"),
        "judge_human_likeness": float("nan"),
        "n": n,
    }

    return auto, sty


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    files = sorted(RESP_DIR.glob("*.csv"))
    files = [f for f in files if "metadata" not in f.name.lower()]
    print(f"Found {len(files)} Qasper response CSVs.", flush=True)

    auto_rows, sty_rows = [], []
    for i, fp in enumerate(files):
        t0 = time.time()
        print(f"  [{i+1}/{len(files)}] {fp.name} …", end=" ", flush=True)
        try:
            auto, sty = process_file(fp)
            if auto:
                auto_rows.append(auto)
            if sty:
                sty_rows.append(sty)
            print(f"done ({time.time()-t0:.1f}s)", flush=True)
        except Exception as exc:
            print(f"ERROR: {exc}", flush=True)

    if auto_rows:
        out = OUT_DIR / "qasper_automatic_metrics.csv"
        pd.DataFrame(auto_rows).to_csv(out, index=False)
        print(f"✓ Saved → {out}", flush=True)
    if sty_rows:
        out = OUT_DIR / "qasper_stylistic_metrics.csv"
        pd.DataFrame(sty_rows).to_csv(out, index=False)
        print(f"✓ Saved → {out}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
