"""
inference/compute_qasper_metrics.py
Tasks 6 & 7 — Compute all automatic and stylistic metrics for Qasper responses.

Automatic metrics  : BLEU-1/2/3/4, ROUGE-1/2/L, METEOR, BERTScore
Stylistic metrics  : Relevance (sentence-transformer cosine), Fluency (GPT-2
                     perplexity), Formality (MTLD), Readability (Flesch),
                     Correctness (SpaCy NER + FuzzyWuzzy)
GPT-4o-as-judge    : 100 random pairs rated on correctness/relevance/
                     human_likeness/demographic_fit (1-5 scale)

Outputs:
    results/qasper_automatic_metrics.csv
    results/qasper_stylistic_metrics.csv

Run:
    cd /path/to/SE_LLM_EVAL
    python -m inference.compute_qasper_metrics
    python -m inference.compute_qasper_metrics --no-judge   # skip GPT-4o judge
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure NLTK data available
for pkg in ("punkt", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

REPO    = Path(__file__).parent.parent
RESP_DIR = REPO / "Qasper_analysis" / "responses"
OUT_DIR  = REPO / "results"
OUT_DIR.mkdir(exist_ok=True)
SEED = 42


# ── Automatic metrics ────────────────────────────────────────────────────────

def compute_bleu(references: list[str], candidate: str) -> dict[str, float]:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    sf = SmoothingFunction().method1
    ref_tokens = [nltk.word_tokenize(r.lower()) for r in references]
    cand_tokens = nltk.word_tokenize(candidate.lower())
    return {
        "bleu_1": sentence_bleu(ref_tokens, cand_tokens, weights=(1, 0, 0, 0), smoothing_function=sf),
        "bleu_2": sentence_bleu(ref_tokens, cand_tokens, weights=(.5, .5, 0, 0), smoothing_function=sf),
        "bleu_3": sentence_bleu(ref_tokens, cand_tokens, weights=(.33, .33, .33, 0), smoothing_function=sf),
        "bleu_4": sentence_bleu(ref_tokens, cand_tokens, weights=(.25, .25, .25, .25), smoothing_function=sf),
    }


def compute_rouge(references: list[str], candidate: str) -> dict[str, float]:
    from rouge import Rouge
    rouge = Rouge()
    scores = {"rouge_1_f": 0, "rouge_2_f": 0, "rouge_l_f": 0}
    best = {"rouge_1_f": 0, "rouge_2_f": 0, "rouge_l_f": 0}
    for ref in references:
        try:
            s = rouge.get_scores(candidate, ref)[0]
            best["rouge_1_f"] = max(best["rouge_1_f"], s["rouge-1"]["f"])
            best["rouge_2_f"] = max(best["rouge_2_f"], s["rouge-2"]["f"])
            best["rouge_l_f"] = max(best["rouge_l_f"], s["rouge-l"]["f"])
        except Exception:
            pass
    return best


def compute_meteor(references: list[str], candidate: str) -> float:
    from nltk.translate.meteor_score import meteor_score
    ref_tokens = [nltk.word_tokenize(r.lower()) for r in references]
    cand_tokens = nltk.word_tokenize(candidate.lower())
    return float(meteor_score(ref_tokens, cand_tokens))


def compute_bertscore(candidates: list[str], references: list[str]) -> dict[str, float]:
    """Batch BERTScore — more efficient than per-example."""
    from bert_score import score as bscore
    P, R, F1 = bscore(
        candidates, references,
        model_type="microsoft/deberta-xlarge-mnli",
        lang="en", verbose=False, batch_size=16,
    )
    return {
        "bertscore_p": float(P.mean()),
        "bertscore_r": float(R.mean()),
        "bertscore_f1": float(F1.mean()),
    }


# ── Stylistic metrics ────────────────────────────────────────────────────────

class StyleScorer:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import spacy

        self._st = SentenceTransformer("all-mpnet-base-v2")
        self._gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
        self._gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
        self._gpt2.eval()
        try:
            self._nlp = spacy.load("en_core_web_sm")
        except OSError:
            os.system("python -m spacy download en_core_web_sm")
            self._nlp = spacy.load("en_core_web_sm")

    def relevance(self, reference: str, candidate: str) -> float:
        from sentence_transformers import util
        a = self._st.encode(reference, convert_to_tensor=True)
        b = self._st.encode(candidate, convert_to_tensor=True)
        return float(util.pytorch_cos_sim(a, b).item())

    def fluency(self, text: str) -> float:
        import torch
        inputs = self._gpt2_tok(text, return_tensors="pt")
        with torch.no_grad():
            loss = self._gpt2(**inputs, labels=inputs["input_ids"]).loss
        return float(torch.exp(loss).item())

    def formality(self, text: str) -> float:
        from lexical_diversity import lex_div as ld
        try:
            tokens = ld.flemmatize(text)
            return float(ld.mtld(tokens)) if tokens else 0.0
        except Exception:
            return 0.0

    def readability(self, text: str) -> float:
        import textstat
        return float(textstat.flesch_reading_ease(text))

    def correctness(self, reference: str, candidate: str) -> float:
        from fuzzywuzzy import fuzz
        ref_doc  = self._nlp(reference)
        cand_doc = self._nlp(candidate)
        ref_ents  = [e.text for e in ref_doc.ents]
        cand_ents = [e.text for e in cand_doc.ents]
        if not ref_ents or not cand_ents:
            return fuzz.token_set_ratio(reference, candidate) / 100
        matches = sum(
            1 for re in ref_ents
            if any(fuzz.token_set_ratio(re, ce) >= 70 for ce in cand_ents)
        )
        return matches / len(ref_ents)


# ── GPT-4o-as-judge ───────────────────────────────────────────────────────────

def gpt4o_judge(
    questions: list[str],
    gold_answers: list[str],
    responses: list[str],
    n_sample: int = 100,
) -> list[dict]:
    from inference.llm_client import LLMClient
    import json as _json

    client = LLMClient("gpt-4o")
    rng = random.Random(SEED)
    indices = rng.sample(range(len(questions)), min(n_sample, len(questions)))
    results = []

    for i in tqdm(indices, desc="GPT-4o judge"):
        system = "You are an expert evaluator. Rate the candidate answer on four dimensions."
        prompt = (
            f"Question: {questions[i]}\n"
            f"Reference Answer: {gold_answers[i]}\n"
            f"Candidate Answer: {responses[i]}\n\n"
            "Rate on 1-5 each:\n"
            "1. Correctness (factually accurate given reference)\n"
            "2. Relevance (answers the question)\n"
            "3. Human-likeness (sounds like a natural human answer)\n"
            "4. Demographic_fit (not applicable here, rate 3)\n\n"
            'Return ONLY JSON: {"correctness": X, "relevance": X, "human_likeness": X, "demographic_fit": X}'
        )
        try:
            raw = client.generate(system, prompt)
            parsed = _json.loads(raw)
        except Exception:
            parsed = {"correctness": None, "relevance": None, "human_likeness": None, "demographic_fit": None}
        results.append({"index": i, **parsed})

    return results


# ── Process one CSV ───────────────────────────────────────────────────────────

def process_file(
    csv_path: Path,
    style: StyleScorer,
    run_judge: bool = True,
) -> tuple[dict, dict]:
    """Returns (auto_metrics_row, style_metrics_row) averaged over all QA pairs."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return {}, {}

    candidates = df["response"].fillna("").tolist()
    # gold_answer may contain multiple references separated by " ||| "
    references = [g.split(" ||| ") for g in df["gold_answer"].fillna("").tolist()]
    single_refs = [r[0] for r in references]
    questions   = df["question"].fillna("").tolist()

    model    = df["model"].iloc[0]  if "model"    in df.columns else str(csv_path.stem)
    prompt   = df["prompt_num"].iloc[0] if "prompt_num" in df.columns else 0
    shot     = df["shot_type"].iloc[0]  if "shot_type"  in df.columns else ""
    year     = df["year"].iloc[0]       if "year"       in df.columns else 2026

    # ── Automatic ──────────────────────────────────────────────────────────
    bleus = [compute_bleu(r, c) for r, c in zip(references, candidates)]
    rouges = [compute_rouge(r, c) for r, c in zip(references, candidates)]
    meteors = [compute_meteor(r, c) for r, c in zip(references, candidates)]

    bs = compute_bertscore(candidates, single_refs)

    auto = {
        "model": model, "prompt_num": prompt, "shot_type": shot, "year": year,
        "bleu_1":   np.mean([b["bleu_1"] for b in bleus]),
        "bleu_2":   np.mean([b["bleu_2"] for b in bleus]),
        "bleu_3":   np.mean([b["bleu_3"] for b in bleus]),
        "bleu_4":   np.mean([b["bleu_4"] for b in bleus]),
        "rouge_1_f": np.mean([r["rouge_1_f"] for r in rouges]),
        "rouge_2_f": np.mean([r["rouge_2_f"] for r in rouges]),
        "rouge_l_f": np.mean([r["rouge_l_f"] for r in rouges]),
        "meteor":   np.mean(meteors),
        **bs,
        "n": len(df),
    }

    # ── Stylistic ──────────────────────────────────────────────────────────
    rel   = [style.relevance(q, c)  for q, c in zip(questions,  candidates)]
    ca_rel = [style.relevance(g, c) for g, c in zip(single_refs, candidates)]
    flu   = [style.fluency(c)       for c in candidates]
    form  = [style.formality(c)     for c in candidates]
    read  = [style.readability(c)   for c in candidates]
    corr  = [style.correctness(g, c) for g, c in zip(single_refs, candidates)]

    # GPT-4o judge
    judge_scores: dict[str, float] = {}
    if run_judge:
        judge = gpt4o_judge(questions, single_refs, candidates)
        for dim in ("correctness", "relevance", "human_likeness"):
            vals = [j[dim] for j in judge if j.get(dim) is not None]
            judge_scores[f"judge_{dim}"] = float(np.mean(vals)) if vals else float("nan")

    sty = {
        "model": model, "prompt_num": prompt, "shot_type": shot, "year": year,
        "qa_relevance":  float(np.mean(rel)),
        "ca_relevance":  float(np.mean(ca_rel)),
        "fluency_ppl":   float(np.mean(flu)),
        "formality_mtld": float(np.mean(form)),
        "readability_flesch": float(np.mean(read)),
        "correctness_ner": float(np.mean(corr)),
        **judge_scores,
        "n": len(df),
    }
    return auto, sty


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip GPT-4o-as-judge (faster, no cost)")
    args = parser.parse_args()

    files = sorted(RESP_DIR.glob("*.csv"))
    if not files:
        print(f"No CSVs found in {RESP_DIR}. Run inference.run_qasper first.")
        sys.exit(1)

    print(f"Found {len(files)} response CSVs. Initialising style scorer …")
    style = StyleScorer()

    auto_rows, sty_rows = [], []
    for fp in tqdm(files, desc="Processing files"):
        try:
            auto, sty = process_file(fp, style, run_judge=not args.no_judge)
            if auto:
                auto_rows.append(auto)
            if sty:
                sty_rows.append(sty)
        except Exception as exc:
            print(f"  ERROR {fp.name}: {exc}")

    if auto_rows:
        pd.DataFrame(auto_rows).to_csv(OUT_DIR / "qasper_automatic_metrics.csv", index=False)
        print(f"✓ Saved → results/qasper_automatic_metrics.csv")
    if sty_rows:
        pd.DataFrame(sty_rows).to_csv(OUT_DIR / "qasper_stylistic_metrics.csv", index=False)
        print(f"✓ Saved → results/qasper_stylistic_metrics.csv")


if __name__ == "__main__":
    main()
