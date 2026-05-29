"""
inference/run_qasper.py
Task 5 — Qasper inference for new models.

Loads the allenai/qasper dataset from HuggingFace, uses the same 50 papers
as the original (seed=42 sample), runs each model × 4 prompt variants,
saves to Qasper_analysis/responses_{model_slug}_{prompt}.csv.

Run:
    cd /path/to/SE_LLM_EVAL
    python -m inference.run_qasper --models gpt-4o claude-sonnet-4-6 llama-3.1-70b
    python -m inference.run_qasper --dry-run
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Optional

import tiktoken
from tqdm import tqdm

REPO    = Path(__file__).parent.parent
OUT_DIR = REPO / "Qasper_analysis" / "responses"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PAPERS   = 50
SEED       = 42
GPT4O_CTX  = 100_000   # token budget for full paper context
MAX_TOKENS_RESPONSE = 256

# ── Prompt builders ───────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLE = (
    "Contents: 'We collected 220 human-human anti-scam dialogs. "
    "The average conversation length is 12.45 turns.'\n"
    "Question: How big is the ANTISCAM dataset?\n"
    "Answer: 220 human-human dialogs\n\n"
)


def _build_messages(
    contents: str,
    question: str,
    prompt_num: int,
    shot_type: str = "few",
) -> tuple[str, str]:
    """Returns (system, user) for the given prompt configuration."""
    few = FEW_SHOT_EXAMPLE if shot_type == "few" else ""

    if prompt_num == 1:   # Expert NLP extractive
        system = (
            "You are a NLP domain expert question answering system. "
            "The user will provide you with the contents of a research paper and a question. "
            "Strictly answer in one sentence or less. "
            "The answer should be an exact extracted text from the contents when possible. "
            'Reply in the format: "answer"'
        )
        user = f"{few}Contents of the research paper: {contents}\n\nQuestion: {question}"

    elif prompt_num == 2: # Plain QA
        system = (
            "You are a question-answering system. "
            "Answer the question using the contents provided. "
            'Reply in the format: "answer"'
        )
        user = f"{few}Contents: {contents}\n\nQuestion: {question}"

    elif prompt_num == 3: # Concise
        system = (
            "The user will provide you with the contents of a research paper and a question. "
            "Answer the question based on the contents in 1 sentence."
        )
        user = f"{few}Contents: {contents}\n\nQuestion: {question}"

    elif prompt_num == 4: # Novice/child persona (best in original paper)
        system = (
            "You are a child who has to answer a question with no knowledge of the domain. "
            "The user will provide you with the contents of a research paper and a question. "
            "Strictly answer in one sentence or less. "
            "The answer should be an exact extracted text from the contents if possible, "
            "otherwise answer like a child reading the contents. "
            'Reply in the format: "answer"'
        )
        user = f"{few}Contents: {contents}\n\nQuestion: {question}"

    else:
        raise ValueError(f"Unknown prompt_num: {prompt_num}")

    return system, user


# ── Token counting ────────────────────────────────────────────────────────────

_ENC = None

def _count_tokens(text: str, model: str = "gpt-4o") -> int:
    global _ENC
    if _ENC is None:
        try:
            _ENC = tiktoken.encoding_for_model(model)
        except Exception:
            _ENC = tiktoken.get_encoding("cl100k_base")
    return len(_ENC.encode(text))


# ── Load Qasper ───────────────────────────────────────────────────────────────

def load_qasper_papers(n: int = N_PAPERS, seed: int = SEED) -> list[dict]:
    """Load and sample n papers from allenai/qasper (HuggingFace)."""
    from datasets import load_dataset
    ds = load_dataset("allenai/qasper", split="test", trust_remote_code=True)
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), n)
    papers = []
    for idx in sorted(indices):
        paper = ds[idx]
        # Build the full text from sections
        full_text_parts = []
        for para in paper.get("full_text", {}).get("paragraphs", []):
            if isinstance(para, list):
                full_text_parts.extend(para)
            else:
                full_text_parts.append(str(para))
        full_text = " ".join(full_text_parts)

        for qa in paper.get("qas", []):
            answers = []
            evidence_parts = []
            for ann in qa.get("answers", []):
                a = ann.get("answer", {})
                spans = a.get("extractive_spans", [])
                free  = a.get("free_form_answer", "")
                evid  = a.get("evidence", [])
                if spans:
                    answers.extend(spans)
                elif free:
                    answers.append(free)
                evidence_parts.extend(evid)

            if not answers:
                continue
            evidence = " ".join(evidence_parts)
            papers.append({
                "paper_id":   paper.get("id", str(idx)),
                "question":   qa["question"],
                "gold_answers": list(set(answers)),
                "evidence":   evidence,
                "full_text":  full_text,
            })
    return papers


# ── Main runner ───────────────────────────────────────────────────────────────

def run_combination(
    model_name: str,
    prompt_num: int,
    shot_type: str,
    papers: list[dict],
    dry_run: bool = False,
) -> Path:
    from inference.llm_client import LLMClient

    slug = model_name.replace("-", "").replace(".", "")[:12]
    out_csv  = OUT_DIR / f"{slug}_p{prompt_num}_{shot_type}.csv"
    out_meta = OUT_DIR / f"{slug}_p{prompt_num}_{shot_type}_metadata.json"

    if out_csv.exists():
        print(f"  SKIP (exists): {out_csv.name}")
        return out_csv
    if dry_run:
        print(f"  DRY RUN: {out_csv.name}")
        return out_csv

    client = LLMClient(model_name, max_tokens=MAX_TOKENS_RESPONSE)
    use_full = model_name in ("gpt-4o",)   # only GPT-4o has large enough context

    meta = client.metadata() | {
        "prompt_num": prompt_num, "shot_type": shot_type, "n_papers": len(papers),
    }

    fieldnames = ["paper_id", "question", "gold_answer", "model", "slug",
                  "prompt_num", "shot_type", "context_type", "response", "year"]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for paper in tqdm(papers, desc=f"{slug} P{prompt_num} {shot_type}"):
        # Decide context
        if use_full and _count_tokens(paper["full_text"]) < GPT4O_CTX:
            context      = paper["full_text"]
            context_type = "full"
        elif paper["evidence"].strip():
            context      = paper["evidence"]
            context_type = "evidence"
        else:
            context      = paper["full_text"][:4000]
            context_type = "truncated"

        system, user = _build_messages(context, paper["question"], prompt_num, shot_type)
        try:
            response = client.generate(system, user)
        except Exception as exc:
            print(f"\n  ERROR: {exc}")
            response = "ERROR"

        row = {
            "paper_id":    paper["paper_id"],
            "question":    paper["question"],
            "gold_answer": " ||| ".join(paper["gold_answers"]),
            "model":       model_name,
            "slug":        slug,
            "prompt_num":  prompt_num,
            "shot_type":   shot_type,
            "context_type": context_type,
            "response":    response,
            "year":        2026,
        }
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ Saved → {out_csv.name}")
    return out_csv


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(SEED)

    parser = argparse.ArgumentParser(description="Qasper inference for new models")
    parser.add_argument("--models", nargs="+",
                        default=["gpt-4o", "claude-sonnet-4-6", "llama-3.1-70b"])
    parser.add_argument("--prompts", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--shots", nargs="+", default=["few"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        print("Loading Qasper dataset …")
        papers = load_qasper_papers()
        print(f"  Loaded {len(papers)} QA pairs from {N_PAPERS} papers.")
    else:
        papers = []

    combos = [
        (m, p, s)
        for m in args.models
        for p in args.prompts
        for s in args.shots
    ]
    print(f"\nPlan: {len(combos)} combinations × ~{len(papers) if papers else N_PAPERS} QA pairs")
    for m, p, s in combos:
        print(f"  {m}  P{p}  {s}-shot")

    if args.dry_run:
        print("\n[dry-run] No API calls made.")
        return

    for model, p, shot in combos:
        print(f"\n→ {model}  P{p}  {shot}-shot")
        run_combination(model, p, shot, papers, dry_run=False)

    print("\n✓ Qasper inference complete.")


if __name__ == "__main__":
    main()
