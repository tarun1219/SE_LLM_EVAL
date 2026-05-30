"""
inference/run_statistics.py
Task 8 — Full statistical analysis on questionnaire responses (2023 + 2026).

Runs all 5 tests:
  1. Independent-samples t-test (each model vs human, per demographic)
  2. Chi-square test (each model vs human response distributions)
  3. Cramér's V
  4. Fleiss' Kappa (per respondent type)
  5. One-way ANOVA (across all respondent types)

Also runs longitudinal comparisons:
  GPT-3.5 → GPT-4o    (same questions, aligned by option_ids)
  LLaMA-2 → LLaMA-3.1

Saves results/statistical_analysis_2026.json.

Run:
    cd /path/to/SE_LLM_EVAL
    python -m inference.run_statistics
"""

from __future__ import annotations
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, ttest_ind
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

REPO    = Path(__file__).parent.parent
DATA    = REPO / "Datasets"
LLM_DIR = REPO / "LLM_Responses"
OUT     = REPO / "results" / "statistical_analysis_2026.json"
SEED    = 42

ANOVA_QUESTIONS = [
    "What is the biggest challenge you face as a developer?",
    "When choosing a programming language for a new project",
    "How do you balance between innovation and meeting project deadlines",
]
DEMO_COLS = ["Age", "Gender", "Experience"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _answer_col(df: pd.DataFrame) -> str:
    return "Option" if "Option" in df.columns else "Answer"


def fleiss_kappa(df: pd.DataFrame) -> float:
    data = df.to_dict("list")
    responses = np.array(list(data.values()))
    option_map: dict = defaultdict(lambda: len(option_map))
    for row in responses:
        for opt in row:
            option_map[opt]
    numeric = np.array([[option_map[opt] for opt in row] for row in responses])
    observed = np.mean(
        np.sum(numeric == np.max(numeric, axis=1, keepdims=True), axis=0)
        / numeric.shape[0]
    )
    def term(i):
        return np.mean(
            (numeric == i)[:, np.newaxis]
            * np.sum(numeric == i, axis=1, keepdims=True)
            / (numeric.shape[1] ** 2)
        )
    chance = sum(term(i) for i in range(len(option_map)))
    return float((observed - chance) / (1 - chance)) if (1 - chance) else 0.0


def ttest_per_demo(df: pd.DataFrame) -> dict:
    le = LabelEncoder()
    col = _answer_col(df)
    df = df.copy()
    df["_enc"] = le.fit_transform(df[col].astype(str))
    results = {}
    for demo in DEMO_COLS:
        if demo not in df.columns:
            continue
        vals = df[demo].dropna().unique()
        if len(vals) < 2:
            continue
        g1, g2 = vals[0], vals[1]
        a = df[df[demo] == g1]["_enc"].values
        b = df[df[demo] == g2]["_enc"].values
        if len(a) < 2 or len(b) < 2:
            continue
        t, p = ttest_ind(a, b)
        sp = np.sqrt(((len(a)-1)*a.std()**2 + (len(b)-1)*b.std()**2) / (len(a)+len(b)-2))
        d = float((a.mean()-b.mean()) / sp) if sp > 0 else 0.0
        results[demo] = {
            f"{g1}_vs_{g2}": {
                "t": round(float(t), 4), "p": round(float(p), 4),
                "significant_p05": bool(p < 0.05), "cohens_d": round(d, 4),
            }
        }
    return results


def chi_sq_cramers(df_model: pd.DataFrame, df_human: pd.DataFrame) -> dict:
    """Chi-square and Cramér's V: model vs human using profile option_id counts."""
    def _get_counts(df):
        col = "Option IDs" if "Option IDs" in df.columns else _answer_col(df)
        return df[col].str.get_dummies(sep=", ").sum()

    mc = _get_counts(df_model)
    hc = _get_counts(df_human)
    common = mc.index.union(hc.index)
    mc = mc.reindex(common, fill_value=0)
    hc = hc.reindex(common, fill_value=0)
    mask = (mc + hc) > 0
    table = np.array([mc[mask].values, hc[mask].values])
    try:
        chi2, p, dof, _ = chi2_contingency(table)
    except Exception:
        return {}
    n = int(table.sum())
    k = int(min(table.shape))
    v = float(np.sqrt(chi2 / (n * (k - 1)))) if k > 1 else 0.0
    return {"chi2": round(float(chi2), 4), "p": round(float(p), 6),
            "dof": dof, "n": n, "cramers_v": round(v, 4),
            "significant_p05": bool(p < 0.05)}


def anova_tests(dfs: dict[str, pd.DataFrame]) -> dict:
    results = {}
    for q in ANOVA_QUESTIONS:
        le = LabelEncoder()
        all_ans, per_group = [], {}
        for label, df in dfs.items():
            col = _answer_col(df)
            mask = df["Question"].str.contains(q, case=False, na=False)
            vals = df.loc[mask, col].dropna().astype(str).tolist()
            per_group[label] = vals
            all_ans.extend(vals)
        if not all_ans:
            continue
        le.fit(all_ans)
        groups = [le.transform(v) for v in per_group.values() if v]
        if len(groups) >= 2:
            F, p = f_oneway(*groups)
            results[q] = {"F": round(float(F), 4), "p": round(float(p), 6),
                          "significant_p05": bool(p < 0.05)}
    return results


# ── Load data ─────────────────────────────────────────────────────────────────

def _load_all_responses() -> dict[str, pd.DataFrame]:
    """Load 2023 baselines + any 2026 response CSVs present."""
    dfs: dict[str, pd.DataFrame] = {}

    # 2023 baselines
    dfs["human"]   = pd.read_csv(DATA / "survey_responses.csv")
    dfs["gpt35"]   = pd.read_csv(DATA / "gpt3.5_responses.csv")
    dfs["llama2"]  = pd.read_csv(DATA / "llama2_responses.csv")

    # 2023 profile-level (for chi-square)
    dfs["gpt35_profile"]  = pd.read_csv(LLM_DIR / "gpt3.5_profile_responses.csv")
    dfs["llama2_profile"] = pd.read_csv(LLM_DIR / "llama2_profile_responses.csv")

    # 2026 — load best prompt variant (P4 few-shot) if available
    for model_slug, label in [
        ("gpt4o",         "gpt4o"),
        ("claudesonnet",  "claude"),   # actual slug is claudesonnet (not claudesonnet46)
        ("llama3370b",    "llama31"),  # actual slug is llama3370b (not llama3170b)
    ]:
        p4_few = LLM_DIR / f"{model_slug}_p4_few_responses.csv"
        if p4_few.exists():
            df26 = pd.read_csv(p4_few)
            # Normalise column names to match 2023 baseline expectations
            col_map = {}
            for c in df26.columns:
                if c.lower() == "answer":
                    col_map[c] = "Answer"
                elif c.lower() == "option_ids":
                    col_map[c] = "Option IDs"
                elif c.lower() == "age":
                    col_map[c] = "Age"
                elif c.lower() == "gender":
                    col_map[c] = "Gender"
                elif c.lower() == "experience":
                    col_map[c] = "Experience"
                elif c.lower() == "question":
                    col_map[c] = "Question"
            if col_map:
                df26 = df26.rename(columns=col_map)
            dfs[label] = df26
            print(f"  Loaded 2026: {label} from {p4_few.name} ({len(df26)} rows)")
        else:
            print(f"  [missing] {p4_few.name} — run inference.run_questionnaire first")

    return dfs


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 62)
    print("TASK 8 — STATISTICAL ANALYSIS 2023 + 2026")
    print("=" * 62)

    dfs = _load_all_responses()
    human_df = dfs["human"]

    results: dict = {
        "generated": __import__("datetime").datetime.now().isoformat(),
        "fleiss_kappa": {},
        "ttest":        {},
        "chi_sq":       {},
        "anova":        {},
        "longitudinal": {},
    }

    # ── Fleiss' Kappa ─────────────────────────────────────────────────────
    print("\nFleiss' Kappa:")
    for label, df in dfs.items():
        if "profile" in label:
            continue
        k = fleiss_kappa(df)
        results["fleiss_kappa"][label] = round(k, 5)
        print(f"  {label:12s}: {k:.5f}")

    # ── t-tests (demographic) ────────────────────────────────────────────
    print("\nDemographic t-tests:")
    for label, df in dfs.items():
        if "profile" in label:
            continue
        t = ttest_per_demo(df)
        results["ttest"][label] = t
        for demo, r in t.items():
            pair = list(r.keys())[0]
            p = r[pair]["p"]
            print(f"  {label:12s} {demo:12s}: p={p:.4f} ({'*' if p<0.05 else ' '})")

    # ── Chi-square + Cramér's V ──────────────────────────────────────────
    print("\nChi-square + Cramér's V (model vs human):")
    profile_pairs = {
        "gpt35":  ("gpt35_profile",  None),
        "llama2": ("llama2_profile", None),
    }
    # Add 2026 models if profile CSVs exist
    for model_slug, label in [("gpt4o","gpt4o"),("claudesonnet","claude"),("llama3370b","llama31")]:
        p4 = LLM_DIR / f"{model_slug}_p4_few_responses.csv"
        if p4.exists():
            profile_pairs[label] = (label, None)

    for label, (prof_key, _) in profile_pairs.items():
        if prof_key not in dfs:
            continue
        cs = chi_sq_cramers(dfs[prof_key], dfs.get("gpt35_profile", dfs["gpt35_profile"]))
        results["chi_sq"][label] = cs
        print(f"  {label:12s}: chi2={cs.get('chi2','n/a')}  p={cs.get('p','n/a')}  V={cs.get('cramers_v','n/a')}")

    # ── ANOVA ────────────────────────────────────────────────────────────
    print("\nANOVA (all respondents):")
    combined = {k: v for k, v in dfs.items() if "profile" not in k}
    anov = anova_tests(combined)
    results["anova"] = anov
    for q, r in anov.items():
        print(f"  {q[:55]}: F={r['F']}  p={r['p']} {'*' if r['significant_p05'] else ' '}")

    # ── Longitudinal comparison ───────────────────────────────────────────
    print("\nLongitudinal comparison:")
    def _longitudinal(old_label, new_label, metric_fn, metric_name):
        if old_label not in dfs or new_label not in dfs:
            return {"note": f"missing {new_label}"}
        old_val = metric_fn(dfs[old_label])
        new_val = metric_fn(dfs[new_label])
        delta = round(new_val - old_val, 5)
        improved = delta > 0
        print(f"  {old_label}→{new_label}: {metric_name} {old_val:.4f}→{new_val:.4f}  Δ={delta:+.4f} {'↑' if improved else '↓'}")
        return {"old": round(old_val,5), "new": round(new_val,5), "delta": delta, "improved": improved}

    results["longitudinal"]["kappa_gpt35_to_gpt4o"]    = _longitudinal("gpt35", "gpt4o", fleiss_kappa, "Fleiss Kappa")
    results["longitudinal"]["kappa_llama2_to_llama31"]  = _longitudinal("llama2", "llama31", fleiss_kappa, "Fleiss Kappa")

    # Demographic bias delta (% significant t-tests)
    def _bias_score(label: str) -> float:
        t = results["ttest"].get(label, {})
        if not t:
            return float("nan")
        sigs = [list(r.values())[0]["significant_p05"] for r in t.values()]
        return float(np.mean(sigs)) if sigs else float("nan")

    for old, new in [("gpt35","gpt4o"), ("llama2","llama31")]:
        if new in dfs:
            o, n = _bias_score(old), _bias_score(new)
            delta = round(n - o, 4) if not np.isnan(o) and not np.isnan(n) else None
            results["longitudinal"][f"bias_{old}_to_{new}"] = {"old": o, "new": n, "delta": delta}
            print(f"  {old}→{new}: bias_score {o:.2f}→{n:.2f}  Δ={delta}")

    # ── Save ──────────────────────────────────────────────────────────────
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved → {OUT}")
    print("=" * 62)


if __name__ == "__main__":
    main()
