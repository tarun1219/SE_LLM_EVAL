"""
inference/run_entropy_analysis.py

Computes Shannon response-entropy per model and question.

Purpose: independent triangulation of the "diversity collapse" finding,
complementing Fleiss' Kappa without the column-wise methodology artefact.

Entropy is defined here as the Shannon entropy of the distribution of
selected answer options across the 10 synthetic demographic profiles,
for each survey question:

    H(q, m) = -Σ_i  p_i · log2(p_i)

where p_i = proportion of profiles that chose option i for question q
under model m.

High entropy → diverse responses across profiles (human-like).
Low entropy  → uniform responses (RLHF-induced homogenisation).

Also computes per-model mean entropy (average over 12 questions)
and profiles-vs-questions entropy heatmap data.

Outputs
-------
results/entropy_analysis.csv  — per-model × per-question entropy table
results/entropy_summary.csv   — per-model mean / median / min / max
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

REPO     = Path(__file__).parent.parent
LLM_DIR  = REPO / "LLM_Responses"
DATA     = REPO / "Datasets"
OUT_DIR  = REPO / "results"
OUT_DIR.mkdir(exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def shannon_entropy(counts: dict) -> float:
    """Shannon entropy (bits) from a dict of option → count."""
    vals = np.array(list(counts.values()), dtype=float)
    vals = vals[vals > 0]
    total = vals.sum()
    if total == 0 or len(vals) == 0:
        return 0.0
    probs = vals / total
    return float(scipy_entropy(probs, base=2))


def max_entropy(n_options: int) -> float:
    """Maximum possible entropy for n_options equally probable choices."""
    if n_options <= 1:
        return 0.0
    return float(np.log2(n_options))


def normalised_entropy(counts: dict) -> float:
    """Entropy divided by maximum possible entropy (range 0–1)."""
    n = len(counts)
    mx = max_entropy(n)
    return shannon_entropy(counts) / mx if mx > 0 else 0.0


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_2026(model_slug: str, prompt: str = "p4_few") -> pd.DataFrame:
    path = LLM_DIR / f"{model_slug}_{prompt}_responses.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    return df[["question", "option_ids"]].dropna()


def load_2023_model(name: str) -> pd.DataFrame:
    """Load 2023 model data (GPT-3.5, LLaMA-2) into (question, option) pairs.
    Expands aggregated rows (Count > 1) into individual profile responses."""
    if name == "gpt35":
        df = pd.read_csv(DATA / "gpt3.5_responses.csv")
    elif name == "llama2":
        df = pd.read_csv(DATA / "llama2_responses.csv")
    else:
        raise ValueError(name)

    rows = []
    if "Count" in df.columns:
        for _, r in df.iterrows():
            for _ in range(int(r.get("Count", 1))):
                rows.append({
                    "question": r["Question"],
                    "option_ids": r.get("Option", r.get("Answer", ""))
                })
    else:
        for _, r in df.iterrows():
            rows.append({
                "question": r["Question"],
                "option_ids": r.get("Answer", "")
            })
    return pd.DataFrame(rows).dropna()


def load_human() -> pd.DataFrame:
    """Human survey — aggregated counts.
    The human CSV has 'Experience' column containing question text (artefact);
    'Question' column is a duplicate. We use 'Question'."""
    df = pd.read_csv(DATA / "survey_responses.csv")
    rows = []
    for _, r in df.iterrows():
        q = r["Question"] if pd.notna(r["Question"]) else r.get("Experience", "")
        opt = r["Option"]
        count = int(r.get("Count", 1))
        for _ in range(count):
            rows.append({"question": q, "option_ids": opt})
    return pd.DataFrame(rows).dropna()


# ── Core computation ─────────────────────────────────────────────────────────

def compute_entropy_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """For each unique question, compute entropy of option distribution."""
    records = []
    for q, grp in df.groupby("question"):
        counts = grp["option_ids"].value_counts().to_dict()
        n_resp = grp["option_ids"].count()
        n_opt  = len(counts)
        H      = shannon_entropy(counts)
        H_norm = normalised_entropy(counts)
        records.append({
            "model":             label,
            "question":          q[:80],   # truncate for readability
            "n_responses":       n_resp,
            "n_distinct_options": n_opt,
            "entropy_bits":      round(H, 4),
            "entropy_norm":      round(H_norm, 4),
            "max_entropy_bits":  round(max_entropy(n_opt), 4),
        })
    return pd.DataFrame(records)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("ENTROPY ANALYSIS — response diversity triangulation")
    print("=" * 60)

    MODELS_2026 = [
        ("gpt4o",        "GPT-4o",        "p4_few"),
        ("claudesonnet", "Claude 4-6",    "p4_few"),
        ("llama3370b",   "LLaMA-3.3-70B", "p4_few"),
        ("gpt4o",        "GPT-4o (P5)",   "p5_zero"),
        ("claudesonnet", "Claude (P5)",   "p5_zero"),
        ("llama3370b",   "LLaMA (P5)",    "p5_zero"),
    ]

    all_tables = []

    # Human
    print("\nHuman baseline...")
    dfh = load_human()
    t = compute_entropy_table(dfh, "Human")
    all_tables.append(t)

    # 2023 baselines
    for slug, label in [("gpt35", "GPT-3.5-Turbo"), ("llama2", "LLaMA-2-7B")]:
        print(f"{label}...")
        df = load_2023_model(slug)
        t = compute_entropy_table(df, label)
        all_tables.append(t)

    # 2026 models
    for slug, label, prompt in MODELS_2026:
        print(f"{label} ({prompt})...")
        df = load_2026(slug, prompt)
        t = compute_entropy_table(df, label)
        all_tables.append(t)

    full = pd.concat(all_tables, ignore_index=True)
    full.to_csv(OUT_DIR / "entropy_analysis.csv", index=False)
    print(f"\n✓ Saved → {OUT_DIR / 'entropy_analysis.csv'}")

    # Summary
    summary_rows = []
    for model, grp in full.groupby("model", sort=False):
        summary_rows.append({
            "model":          model,
            "mean_entropy_bits":   round(grp["entropy_bits"].mean(), 4),
            "median_entropy_bits": round(grp["entropy_bits"].median(), 4),
            "min_entropy_bits":    round(grp["entropy_bits"].min(), 4),
            "max_entropy_bits":    round(grp["entropy_bits"].max(), 4),
            "mean_entropy_norm":   round(grp["entropy_norm"].mean(), 4),
            "n_questions":         len(grp),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "entropy_summary.csv", index=False)
    print(f"✓ Saved → {OUT_DIR / 'entropy_summary.csv'}")

    print("\nMean entropy (bits) per model:")
    print("-" * 55)
    for _, r in summary.iterrows():
        bar = "█" * int(r["mean_entropy_bits"] * 10)
        print(f"  {r['model']:22s}: {r['mean_entropy_bits']:.4f}  {bar}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
