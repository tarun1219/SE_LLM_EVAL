"""
inference/run_categorical_robustness.py

Categorical robustness checks for demographic sensitivity.

Replaces / supplements the label-encoded t-test approach with
category-native tests:

1.  Chi-square test of independence: demographic group × response option,
    per question, per model.
2.  Fisher's exact test (2×2 only): for binary demographic dimensions
    (Gender: Man/Woman).

The t-test approach (label-encoding ordinal answers) is a coarse
approximation; these tests treat response options as categories.

All analyses use archived response CSVs only — no API calls.

Output
------
results/categorical_robustness_tests.csv  — per model × question × demo
results/categorical_robustness_summary.csv — proportion of sig. tests
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

REPO    = Path(__file__).parent.parent
LLM_DIR = REPO / "LLM_Responses"
DATA    = REPO / "Datasets"
OUT_DIR = REPO / "results"
OUT_DIR.mkdir(exist_ok=True)

ALPHA = 0.05


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_2026(model_slug: str, prompt: str = "p4_few") -> pd.DataFrame:
    path = LLM_DIR / f"{model_slug}_{prompt}_responses.csv"
    df = pd.read_csv(path)
    # normalise column names
    rename = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename)
    keep = ["question", "option_ids", "age", "gender", "experience"]
    return df[[c for c in keep if c in df.columns]].dropna(subset=["option_ids"])


def load_human_individual() -> pd.DataFrame:
    """Expand aggregated human counts into individual rows."""
    df = pd.read_csv(DATA / "survey_responses.csv")
    rows = []
    for _, r in df.iterrows():
        q = r["Question"] if pd.notna(r.get("Question")) else r.get("Experience", "")
        cnt = int(r.get("Count", 1))
        for _ in range(cnt):
            rows.append({
                "question":   q,
                "option_ids": r.get("Option", ""),
                "age":        r.get("Age", ""),
                "gender":     r.get("Gender", ""),
                "experience": ""  # not available in human baseline
            })
    return pd.DataFrame(rows)


# ── Test helpers ─────────────────────────────────────────────────────────────

def chi2_demo_vs_response(df: pd.DataFrame, demo_col: str,
                           question: str) -> dict:
    """Chi-square test: demographic group × response option for one question."""
    sub = df[df["question"].str.contains(question, case=False, na=False)]
    if sub.empty or demo_col not in sub.columns:
        return {}
    groups = sub[demo_col].dropna().unique()
    if len(groups) < 2:
        return {}
    options = sub["option_ids"].dropna().unique()
    if len(options) < 2:
        return {"chi2": 0.0, "p": 1.0, "dof": 0, "sig": False,
                "test": "chi2", "note": "only one option observed"}

    # Build contingency table
    table = []
    for g in groups:
        row = []
        for o in options:
            row.append(int(((sub[demo_col] == g) & (sub["option_ids"] == o)).sum()))
        table.append(row)
    table = np.array(table)

    # Drop all-zero columns
    nonzero_cols = table.sum(axis=0) > 0
    table = table[:, nonzero_cols]
    if table.shape[1] < 2:
        return {"chi2": 0.0, "p": 1.0, "dof": 0, "sig": False,
                "test": "chi2", "note": "only one option after filtering"}

    try:
        chi2, p, dof, _ = chi2_contingency(table)
    except Exception as e:
        return {"note": str(e)}

    result = {
        "chi2":  round(float(chi2), 4),
        "p":     round(float(p), 6),
        "dof":   int(dof),
        "sig":   bool(p < ALPHA),
        "test":  "chi2",
        "n_groups": len(groups),
        "n_options": int(nonzero_cols.sum()),
    }

    # Fisher's exact for 2×2
    if table.shape == (2, 2):
        _, pf = fisher_exact(table)
        result["fisher_p"] = round(float(pf), 6)
        result["fisher_sig"] = bool(pf < ALPHA)

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("CATEGORICAL ROBUSTNESS TESTS")
    print("=" * 60)

    MODELS = [
        ("human",        None,           None),
        ("gpt4o",        "GPT-4o",       "p4_few"),
        ("claudesonnet", "Claude 4-6",   "p4_few"),
        ("llama3370b",   "LLaMA-3.3",    "p4_few"),
    ]
    DEMO_COLS = ["age", "gender", "experience"]

    records = []

    for slug, label, prompt in MODELS:
        if slug == "human":
            df = load_human_individual()
            label = "Human"
        else:
            df = load_2026(slug, prompt)

        questions = df["question"].dropna().unique()
        print(f"\n{label}: {len(questions)} questions × {len(DEMO_COLS)} demos")

        for q in questions:
            q_short = q[:60]
            for demo in DEMO_COLS:
                res = chi2_demo_vs_response(df, demo, q)
                if not res:
                    continue
                records.append({
                    "model":      label,
                    "question":   q_short,
                    "demographic": demo,
                    **res
                })
                if res.get("sig"):
                    print(f"  *** SIGNIFICANT: {demo} | {q_short[:40]}  "
                          f"chi2={res.get('chi2'):.2f}  p={res.get('p'):.4f}")

    results_df = pd.DataFrame(records)
    results_df.to_csv(OUT_DIR / "categorical_robustness_tests.csv", index=False)
    print(f"\n✓ Saved → {OUT_DIR / 'categorical_robustness_tests.csv'}")

    # Summary: proportion of significant tests per model × demo
    print("\nProportion of significant chi-square tests (p < 0.05):")
    print("-" * 55)
    summary_rows = []
    for model, grp in results_df.groupby("model"):
        total = len(grp)
        n_sig = grp["sig"].sum()
        row = {"model": model, "n_tests": total,
               "n_significant": int(n_sig),
               "prop_significant": round(float(n_sig / total) if total else 0, 3)}
        for demo in DEMO_COLS:
            sub = grp[grp["demographic"] == demo]
            ns = sub["sig"].sum()
            row[f"prop_{demo}"] = round(
                float(ns / len(sub)) if len(sub) else 0, 3)
        summary_rows.append(row)
        print(f"  {model:22s}: {n_sig}/{total} sig  ({row['prop_significant']:.1%})")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "categorical_robustness_summary.csv", index=False)
    print(f"\n✓ Saved → {OUT_DIR / 'categorical_robustness_summary.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
