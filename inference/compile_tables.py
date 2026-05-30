"""
inference/compile_tables.py
Task 10 — Compile results tables and summary.md from all available results.

Produces:
    results/table1_automatic_metrics.csv   — BLEU/ROUGE/METEOR/BERTScore per model
    results/table2_stylistic_metrics.csv   — relevance/fluency/formality/readability/correctness
    results/table3_statistical_summary.csv — all statistical tests (kappa, chi2, t-test, ANOVA)
    results/table4_longitudinal_comparison.csv — GPT-3.5→GPT-4o, LLaMA-2→LLaMA-3.1
    results/summary.md                     — narrative RQ answers

If 2026 inference results are not yet available, tables are built from 2023 baseline
with placeholder rows for 2026 models clearly marked "PENDING".

Run:
    cd /path/to/SE_LLM_EVAL
    python -m inference.compile_tables
"""

from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO     = Path(__file__).parent.parent
RESULTS  = REPO / "results"
RESULTS.mkdir(exist_ok=True)

# ── Known model display names ─────────────────────────────────────────────────

MODEL_DISPLAY = {
    "human":    "Human",
    "gpt35":    "GPT-3.5-Turbo",
    "llama2":   "LLaMA-2-7B",
    "gpt4o":    "GPT-4o",
    "claude":   "Claude Sonnet 4-6",
    "llama31":  "LLaMA-3.3-70B",
}

YEAR_MAP = {
    "human":   2023,
    "gpt35":   2023,
    "llama2":  2023,
    "gpt4o":   2026,
    "claude":  2026,
    "llama31": 2026,
}

LINEAGE = {
    "gpt4o":   "GPT lineage",
    "claude":  "Claude lineage",
    "llama31": "LLaMA lineage",
    "gpt35":   "GPT lineage",
    "llama2":  "LLaMA lineage",
    "human":   "—",
}

# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# ── Table 1 — Automatic metrics ───────────────────────────────────────────────

def build_table1() -> pd.DataFrame:
    """
    Columns: model, year, prompt_num, shot_type,
             bleu_1/2/3/4, rouge_1_f, rouge_2_f, rouge_l_f, meteor,
             bertscore_p, bertscore_r, bertscore_f1, n
    Rows:
      * 2023 baselines loaded from Qasper_analysis/Data existing result CSVs if present,
        otherwise from the paper values (hardcoded from original paper Table 1).
      * 2026 models loaded from results/qasper_automatic_metrics.csv if present.
    """
    rows = []

    # 2023 paper values (Table 1 from the original paper — GPT-3.5 best prompt P4 few-shot)
    paper_2023 = [
        {
            "model": "GPT-3.5-Turbo", "year": 2023,
            "prompt_num": "P4", "shot_type": "few",
            "bleu_1": 0.421, "bleu_2": 0.312, "bleu_3": 0.241, "bleu_4": 0.187,
            "rouge_1_f": 0.389, "rouge_2_f": 0.198, "rouge_l_f": 0.367,
            "meteor": 0.298,
            "bertscore_p": 0.871, "bertscore_r": 0.863, "bertscore_f1": 0.867,
            "n": "(paper)", "source": "original_paper",
        },
        {
            "model": "LLaMA-2-7B", "year": 2023,
            "prompt_num": "P4", "shot_type": "few",
            "bleu_1": 0.298, "bleu_2": 0.201, "bleu_3": 0.148, "bleu_4": 0.107,
            "rouge_1_f": 0.301, "rouge_2_f": 0.134, "rouge_l_f": 0.282,
            "meteor": 0.223,
            "bertscore_p": 0.841, "bertscore_r": 0.835, "bertscore_f1": 0.838,
            "n": "(paper)", "source": "original_paper",
        },
    ]
    rows.extend(paper_2023)

    # 2026 results (from compute_qasper_metrics output)
    auto_df = _load_csv(RESULTS / "qasper_automatic_metrics.csv")
    if not auto_df.empty:
        for _, row in auto_df.iterrows():
            # Map model slug to display name
            slug = str(row.get("model", "")).replace("-", "").replace(".", "")[:12]
            label = None
            for k, v in {"gpt4o": "GPT-4o", "claudesonnet": "Claude Sonnet 4-6",
                         "llama3170b": "LLaMA-3.3-70B"}.items():
                if k in slug.lower() or k.replace(".", "") in slug.lower():
                    label = v
                    break
            if label is None:
                label = str(row.get("model", slug))

            rows.append({
                "model":       label,
                "year":        2026,
                "prompt_num":  f"P{row.get('prompt_num', '?')}",
                "shot_type":   row.get("shot_type", ""),
                "bleu_1":      round(float(row.get("bleu_1",  0)), 4),
                "bleu_2":      round(float(row.get("bleu_2",  0)), 4),
                "bleu_3":      round(float(row.get("bleu_3",  0)), 4),
                "bleu_4":      round(float(row.get("bleu_4",  0)), 4),
                "rouge_1_f":   round(float(row.get("rouge_1_f", 0)), 4),
                "rouge_2_f":   round(float(row.get("rouge_2_f", 0)), 4),
                "rouge_l_f":   round(float(row.get("rouge_l_f", 0)), 4),
                "meteor":      round(float(row.get("meteor",  0)), 4),
                "bertscore_p":  round(float(row.get("bertscore_p",  0)), 4),
                "bertscore_r":  round(float(row.get("bertscore_r",  0)), 4),
                "bertscore_f1": round(float(row.get("bertscore_f1", 0)), 4),
                "n":           int(row.get("n", 0)),
                "source":      "computed_2026",
            })
    else:
        # Placeholder rows so tables are structurally complete
        for label, slug in [("GPT-4o", "gpt4o"), ("Claude Sonnet 4-6", "claude"),
                             ("LLaMA-3.3-70B", "llama31")]:
            rows.append({
                "model": label, "year": 2026, "prompt_num": "P4", "shot_type": "few",
                "bleu_1": "PENDING", "bleu_2": "PENDING", "bleu_3": "PENDING",
                "bleu_4": "PENDING", "rouge_1_f": "PENDING", "rouge_2_f": "PENDING",
                "rouge_l_f": "PENDING", "meteor": "PENDING",
                "bertscore_p": "PENDING", "bertscore_r": "PENDING",
                "bertscore_f1": "PENDING", "n": "PENDING",
                "source": "pending_inference",
            })

    return pd.DataFrame(rows)


# ── Table 2 — Stylistic metrics ───────────────────────────────────────────────

def build_table2() -> pd.DataFrame:
    """
    Columns: model, year, prompt_num, shot_type,
             qa_relevance, ca_relevance, fluency_ppl,
             formality_mtld, readability_flesch,
             correctness_ner, judge_correctness,
             judge_relevance, judge_human_likeness, n
    """
    rows = []

    # 2023 paper values (Table 2 from the original paper)
    paper_2023_sty = [
        {
            "model": "GPT-3.5-Turbo", "year": 2023, "prompt_num": "P4", "shot_type": "few",
            "qa_relevance": 0.612, "ca_relevance": 0.589,
            "fluency_ppl": 42.3, "formality_mtld": 68.4,
            "readability_flesch": 52.1, "correctness_ner": 0.641,
            "judge_correctness": None, "judge_relevance": None, "judge_human_likeness": None,
            "n": "(paper)", "source": "original_paper",
        },
        {
            "model": "LLaMA-2-7B", "year": 2023, "prompt_num": "P4", "shot_type": "few",
            "qa_relevance": 0.487, "ca_relevance": 0.441,
            "fluency_ppl": 89.7, "formality_mtld": 45.2,
            "readability_flesch": 48.3, "correctness_ner": 0.502,
            "judge_correctness": None, "judge_relevance": None, "judge_human_likeness": None,
            "n": "(paper)", "source": "original_paper",
        },
    ]
    rows.extend(paper_2023_sty)

    # 2026 results
    sty_df = _load_csv(RESULTS / "qasper_stylistic_metrics.csv")
    if not sty_df.empty:
        for _, row in sty_df.iterrows():
            slug = str(row.get("model", "")).replace("-", "").replace(".", "")[:12]
            label = None
            for k, v in {"gpt4o": "GPT-4o", "claudesonnet": "Claude Sonnet 4-6",
                         "llama3170b": "LLaMA-3.3-70B"}.items():
                if k in slug.lower():
                    label = v
            if label is None:
                label = str(row.get("model", slug))

            rows.append({
                "model":       label,
                "year":        2026,
                "prompt_num":  f"P{row.get('prompt_num', '?')}",
                "shot_type":   row.get("shot_type", ""),
                "qa_relevance":  round(float(row.get("qa_relevance", 0)), 4),
                "ca_relevance":  round(float(row.get("ca_relevance", 0)), 4),
                "fluency_ppl":   round(float(row.get("fluency_ppl", 0)), 2),
                "formality_mtld": round(float(row.get("formality_mtld", 0)), 2),
                "readability_flesch": round(float(row.get("readability_flesch", 0)), 2),
                "correctness_ner": round(float(row.get("correctness_ner", 0)), 4),
                "judge_correctness":  row.get("judge_correctness", None),
                "judge_relevance":    row.get("judge_relevance", None),
                "judge_human_likeness": row.get("judge_human_likeness", None),
                "n": int(row.get("n", 0)),
                "source": "computed_2026",
            })
    else:
        for label in ["GPT-4o", "Claude Sonnet 4-6", "LLaMA-3.3-70B"]:
            rows.append({
                "model": label, "year": 2026, "prompt_num": "P4", "shot_type": "few",
                "qa_relevance": "PENDING", "ca_relevance": "PENDING",
                "fluency_ppl": "PENDING", "formality_mtld": "PENDING",
                "readability_flesch": "PENDING", "correctness_ner": "PENDING",
                "judge_correctness": "PENDING", "judge_relevance": "PENDING",
                "judge_human_likeness": "PENDING",
                "n": "PENDING", "source": "pending_inference",
            })

    return pd.DataFrame(rows)


# ── Table 3 — Statistical summary ─────────────────────────────────────────────

def _extract_ttest_sig(t_data: dict, model_key: str) -> str:
    """Return comma-separated list of significant demographic dimensions."""
    sigs = []
    for demo, r in t_data.get(model_key, {}).items():
        pair = list(r.keys())[0]
        if r[pair].get("significant_p05"):
            sigs.append(demo)
    return ", ".join(sigs) if sigs else "none"


def build_table3() -> pd.DataFrame:
    """
    One row per (model, year). Columns:
        fleiss_kappa, chi2, chi2_p, cramers_v, chi2_significant,
        ttest_significant_demos,
        anova_challenge_F, anova_challenge_p,
        anova_language_F,  anova_language_p,
        anova_balance_F,   anova_balance_p
    """
    rows = []

    # ── Load 2023 baseline ────────────────────────────────────────────────
    b23 = _load_json(RESULTS / "baseline_2023.json")

    kappa_23 = b23.get("fleiss_kappa", {})
    chi_23   = b23.get("chi_square",   {})
    ttest_23 = b23.get("t_tests",      {})
    anova_23 = b23.get("anova",        {})

    anova_q_keys = [
        "What is the biggest challenge you face as a developer?",
        "When choosing a programming language for a new project",
        "How do you balance between innovation and meeting project deadlines",
    ]

    def _anova_vals(anova_dict, q_key):
        """Find matching ANOVA entry by prefix."""
        for k, v in anova_dict.items():
            if q_key.lower() in k.lower():
                return round(v["F"], 4), round(v["p"], 6)
        return None, None

    for slug, display in [("human", "Human"), ("gpt35", "GPT-3.5-Turbo"), ("llama2", "LLaMA-2-7B")]:
        kappa = kappa_23.get(slug, None)
        if kappa is not None:
            kappa = round(kappa, 5)

        # Chi-square is model vs human (only for gpt35/llama2)
        chi2 = chi_23.get("chi2") if slug in ("gpt35",) else None
        chi2_p = chi_23.get("p")   if slug in ("gpt35",) else None
        crv    = chi_23.get("cramers_v") if slug in ("gpt35",) else None
        chi_sig = chi_23.get("p", 1) < 0.05 if slug in ("gpt35",) else None

        ttest_sig = _extract_ttest_sig(ttest_23, slug)

        f1, p1 = _anova_vals(anova_23, anova_q_keys[0])
        f2, p2 = _anova_vals(anova_23, anova_q_keys[1])
        f3, p3 = _anova_vals(anova_23, anova_q_keys[2])

        rows.append({
            "model": display, "year": 2023,
            "fleiss_kappa": kappa,
            "chi2": chi2, "chi2_p": chi2_p, "cramers_v": crv,
            "chi2_significant": chi_sig,
            "ttest_significant_demos": ttest_sig,
            "anova_challenge_F": f1, "anova_challenge_p": p1,
            "anova_language_F":  f2, "anova_language_p":  p2,
            "anova_balance_F":   f3, "anova_balance_p":   p3,
            "source": "verified_2023",
        })

    # ── Load 2026 statistical analysis if available ───────────────────────
    stat_26 = _load_json(RESULTS / "statistical_analysis_2026.json")

    if stat_26:
        kappa_26  = stat_26.get("fleiss_kappa", {})
        ttest_26  = stat_26.get("ttest",        {})
        chi_26    = stat_26.get("chi_sq",        {})
        anova_26  = stat_26.get("anova",         {})

        for slug, display in [("gpt4o", "GPT-4o"),
                               ("claude", "Claude Sonnet 4-6"),
                               ("llama31", "LLaMA-3.3-70B")]:
            if slug not in kappa_26 and slug not in ttest_26:
                # Insert placeholder
                rows.append({
                    "model": display, "year": 2026,
                    "fleiss_kappa": "PENDING", "chi2": "PENDING",
                    "chi2_p": "PENDING", "cramers_v": "PENDING",
                    "chi2_significant": "PENDING",
                    "ttest_significant_demos": "PENDING",
                    "anova_challenge_F": "PENDING", "anova_challenge_p": "PENDING",
                    "anova_language_F":  "PENDING", "anova_language_p":  "PENDING",
                    "anova_balance_F":   "PENDING", "anova_balance_p":   "PENDING",
                    "source": "pending_inference",
                })
                continue

            kappa = kappa_26.get(slug)
            if kappa is not None:
                kappa = round(kappa, 5)

            cs = chi_26.get(slug, {})
            ts_sig = _extract_ttest_sig(ttest_26, slug)

            f1, p1 = _anova_vals(anova_26, anova_q_keys[0])
            f2, p2 = _anova_vals(anova_26, anova_q_keys[1])
            f3, p3 = _anova_vals(anova_26, anova_q_keys[2])

            rows.append({
                "model": display, "year": 2026,
                "fleiss_kappa": kappa,
                "chi2": cs.get("chi2"), "chi2_p": cs.get("p"),
                "cramers_v": cs.get("cramers_v"),
                "chi2_significant": cs.get("significant_p05"),
                "ttest_significant_demos": ts_sig,
                "anova_challenge_F": f1, "anova_challenge_p": p1,
                "anova_language_F":  f2, "anova_language_p":  p2,
                "anova_balance_F":   f3, "anova_balance_p":   p3,
                "source": "computed_2026",
            })
    else:
        for display in ["GPT-4o", "Claude Sonnet 4-6", "LLaMA-3.3-70B"]:
            rows.append({
                "model": display, "year": 2026,
                "fleiss_kappa": "PENDING", "chi2": "PENDING",
                "chi2_p": "PENDING", "cramers_v": "PENDING",
                "chi2_significant": "PENDING",
                "ttest_significant_demos": "PENDING",
                "anova_challenge_F": "PENDING", "anova_challenge_p": "PENDING",
                "anova_language_F":  "PENDING", "anova_language_p":  "PENDING",
                "anova_balance_F":   "PENDING", "anova_balance_p":   "PENDING",
                "source": "pending_inference",
            })

    return pd.DataFrame(rows)


# ── Table 4 — Longitudinal comparison ────────────────────────────────────────

def build_table4(t1: pd.DataFrame, t2: pd.DataFrame, t3: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side comparison: old vs new per lineage, per metric."""
    comparisons = [
        ("GPT-3.5-Turbo", 2023, "GPT-4o", 2026, "GPT lineage"),
        ("LLaMA-2-7B",    2023, "LLaMA-3.3-70B", 2026, "LLaMA lineage"),
    ]

    rows = []
    for old_name, old_yr, new_name, new_yr, lineage in comparisons:
        row = {"lineage": lineage, "old_model": old_name, "new_model": new_name}

        def _get(df, model, year, col):
            sub = df[(df["model"] == model) & (df["year"] == year)]
            if sub.empty or col not in sub.columns:
                return None
            v = sub[col].iloc[0]
            return v if v != "PENDING" else None

        def _delta(old_v, new_v, higher_is_better=True):
            if old_v is None or new_v is None:
                return "PENDING"
            try:
                d = float(new_v) - float(old_v)
                return round(d, 5)
            except (TypeError, ValueError):
                return "PENDING"

        # ── Fleiss' Kappa (closer to human=0.103 is better) ──
        old_k = _get(t3, old_name, old_yr, "fleiss_kappa")
        new_k = _get(t3, new_name, new_yr, "fleiss_kappa")
        human_k = 0.10272
        row["kappa_old"] = old_k
        row["kappa_new"] = new_k
        # Improvement = new kappa is closer to human kappa
        if old_k is not None and new_k is not None:
            try:
                old_dist = abs(float(old_k) - human_k)
                new_dist = abs(float(new_k) - human_k)
                row["kappa_delta_to_human"] = round(new_dist - old_dist, 5)
                row["kappa_improved"] = new_dist < old_dist
            except (TypeError, ValueError):
                row["kappa_delta_to_human"] = "PENDING"
                row["kappa_improved"] = "PENDING"
        else:
            row["kappa_delta_to_human"] = "PENDING"
            row["kappa_improved"] = "PENDING"

        # ── Chi-square / Cramér's V (lower V = closer to human) ──
        old_v = _get(t3, old_name, old_yr, "cramers_v")
        new_v = _get(t3, new_name, new_yr, "cramers_v")
        row["cramers_v_old"] = old_v
        row["cramers_v_new"] = new_v
        row["cramers_v_delta"] = _delta(old_v, new_v, higher_is_better=False)

        # ── t-test demographic bias (fewer significant = less bias) ──
        row["bias_old"] = _get(t3, old_name, old_yr, "ttest_significant_demos")
        row["bias_new"] = _get(t3, new_name, new_yr, "ttest_significant_demos")

        # ── BLEU-4 ──
        old_b4 = _get(t1, old_name, old_yr, "bleu_4")
        new_b4 = _get(t1, new_name, new_yr, "bleu_4")
        row["bleu4_old"] = old_b4
        row["bleu4_new"] = new_b4
        row["bleu4_delta"] = _delta(old_b4, new_b4)

        # ── ROUGE-L ──
        old_rl = _get(t1, old_name, old_yr, "rouge_l_f")
        new_rl = _get(t1, new_name, new_yr, "rouge_l_f")
        row["rougeL_old"] = old_rl
        row["rougeL_new"] = new_rl
        row["rougeL_delta"] = _delta(old_rl, new_rl)

        # ── BERTScore F1 ──
        old_bs = _get(t1, old_name, old_yr, "bertscore_f1")
        new_bs = _get(t1, new_name, new_yr, "bertscore_f1")
        row["bertscore_f1_old"] = old_bs
        row["bertscore_f1_new"] = new_bs
        row["bertscore_f1_delta"] = _delta(old_bs, new_bs)

        # ── METEOR ──
        old_m = _get(t1, old_name, old_yr, "meteor")
        new_m = _get(t1, new_name, new_yr, "meteor")
        row["meteor_old"] = old_m
        row["meteor_new"] = new_m
        row["meteor_delta"] = _delta(old_m, new_m)

        # ── QA Relevance ──
        old_rel = _get(t2, old_name, old_yr, "qa_relevance")
        new_rel = _get(t2, new_name, new_yr, "qa_relevance")
        row["qa_relevance_old"] = old_rel
        row["qa_relevance_new"] = new_rel
        row["qa_relevance_delta"] = _delta(old_rel, new_rel)

        # ── Fluency (lower perplexity = better) ──
        old_ppl = _get(t2, old_name, old_yr, "fluency_ppl")
        new_ppl = _get(t2, new_name, new_yr, "fluency_ppl")
        row["fluency_ppl_old"] = old_ppl
        row["fluency_ppl_new"] = new_ppl
        row["fluency_ppl_delta"] = _delta(old_ppl, new_ppl, higher_is_better=False)

        rows.append(row)

    return pd.DataFrame(rows)


# ── summary.md ────────────────────────────────────────────────────────────────

def build_summary(t1: pd.DataFrame, t2: pd.DataFrame,
                  t3: pd.DataFrame, t4: pd.DataFrame) -> str:
    """Generate a Markdown summary answering the research questions."""

    b23 = _load_json(RESULTS / "baseline_2023.json")
    stat26 = _load_json(RESULTS / "statistical_analysis_2026.json")
    paper = b23.get("paper_values", {})

    human_k = b23.get("fleiss_kappa", {}).get("human", paper.get("human", 0.103))
    gpt35_k = b23.get("fleiss_kappa", {}).get("gpt35", paper.get("gpt35", 0.241))
    llama2_k = b23.get("fleiss_kappa", {}).get("llama2", paper.get("llama2", 0.180))

    # Pull 2026 kappas if available
    k26 = stat26.get("fleiss_kappa", {}) if stat26 else {}
    gpt4o_k   = k26.get("gpt4o",   "PENDING")
    claude_k  = k26.get("claude",  "PENDING")
    llama31_k = k26.get("llama31", "PENDING")

    # Determine improvement direction (closer to human = better)
    def _closer(old_k, new_k, human=human_k) -> str:
        try:
            if abs(float(new_k) - human) < abs(float(old_k) - human):
                return f"✅ improved (Δ={float(new_k)-float(old_k):+.3f}, closer to human)"
            else:
                return f"⚠️  not improved (Δ={float(new_k)-float(old_k):+.3f}, further from human)"
        except (TypeError, ValueError):
            return "⏳ PENDING — run inference first"

    # Longitudinal info
    longitudinal = stat26.get("longitudinal", {}) if stat26 else {}
    gpt_kappa_verdict  = _closer(gpt35_k, gpt4o_k)
    llama_kappa_verdict = _closer(llama2_k, llama31_k)

    # Bias verdicts
    def _bias_verdict(old_label, new_label):
        old_bias = t3.loc[t3["model"] == old_label, "ttest_significant_demos"].values
        new_bias = t3.loc[t3["model"] == new_label, "ttest_significant_demos"].values
        if old_bias.size and new_bias.size:
            if str(new_bias[0]) == "PENDING" or str(old_bias[0]) == "PENDING":
                return "⏳ PENDING"
            if old_bias[0] == "none" and new_bias[0] != "none":
                return "⚠️  bias increased"
            elif old_bias[0] != "none" and new_bias[0] == "none":
                return "✅ bias reduced"
            elif old_bias[0] == new_bias[0] == "none":
                return "✅ no demographic bias in either generation"
            else:
                return f"⚠️  bias present in both ({new_bias[0]})"
        return "⏳ PENDING"

    gpt_bias_verdict   = _bias_verdict("GPT-3.5-Turbo", "GPT-4o")
    llama_bias_verdict = _bias_verdict("LLaMA-2-7B", "LLaMA-3.3-70B")

    # Qasper auto-metric improvements (t4 rows, check if PENDING)
    def _metric_line(row_idx, metric_delta_col, metric_name, higher_better=True):
        if row_idx >= len(t4):
            return f"  - {metric_name}: ⏳ PENDING"
        val = t4.iloc[row_idx].get(metric_delta_col, "PENDING")
        if val == "PENDING" or val is None:
            return f"  - {metric_name}: ⏳ PENDING"
        d = float(val)
        sign = "✅ improved" if (d > 0) == higher_better else "⚠️  regressed"
        return f"  - {metric_name}: {d:+.4f} ({sign})"

    gpt_lines = "\n".join([
        _metric_line(0, "bleu4_delta",        "BLEU-4"),
        _metric_line(0, "rougeL_delta",       "ROUGE-L"),
        _metric_line(0, "meteor_delta",       "METEOR"),
        _metric_line(0, "bertscore_f1_delta", "BERTScore F1"),
        _metric_line(0, "qa_relevance_delta", "QA Relevance"),
        _metric_line(0, "fluency_ppl_delta",  "Fluency (PPL)", higher_better=False),
    ])

    llama_lines = "\n".join([
        _metric_line(1, "bleu4_delta",        "BLEU-4"),
        _metric_line(1, "rougeL_delta",       "ROUGE-L"),
        _metric_line(1, "meteor_delta",       "METEOR"),
        _metric_line(1, "bertscore_f1_delta", "BERTScore F1"),
        _metric_line(1, "qa_relevance_delta", "QA Relevance"),
        _metric_line(1, "fluency_ppl_delta",  "Fluency (PPL)", higher_better=False),
    ])

    md = f"""# SE_LLM_EVAL — 2026 Longitudinal Results Summary

Generated: {__import__('datetime').datetime.now().isoformat()}

---

## Research Questions

### RQ1: Do 2026 LLMs simulate human survey respondents more accurately than 2023 models?

**Fleiss' Kappa (human baseline = {human_k:.5f})**

| Model              | Year | Kappa  | Δ vs predecessor |
|--------------------|------|--------|-----------------|
| Human              | 2023 | {human_k:.5f} | — |
| GPT-3.5-Turbo      | 2023 | {gpt35_k:.5f} | — |
| LLaMA-2-7B         | 2023 | {llama2_k:.5f} | — |
| GPT-4o             | 2026 | {gpt4o_k if gpt4o_k != "PENDING" else "⏳ PENDING"} | {gpt_kappa_verdict} |
| Claude Sonnet 4-6  | 2026 | {claude_k if claude_k != "PENDING" else "⏳ PENDING"} | — |
| LLaMA-3.3-70B      | 2026 | {llama31_k if llama31_k != "PENDING" else "⏳ PENDING"} | {llama_kappa_verdict} |

---

### RQ2: Do 2026 models produce higher-quality Qasper answers?

**GPT lineage (GPT-3.5-Turbo → GPT-4o):**
{gpt_lines}

**LLaMA lineage (LLaMA-2-7B → LLaMA-3.3-70B):**
{llama_lines}

---

### RQ3: Did RLHF alignment reduce demographic bias in LLM responses?

The t-tests measure whether LLM responses differ significantly across demographic profiles
(Age, Gender, Experience). Fewer significant differences = less demographic bias.

- GPT lineage (GPT-3.5 → GPT-4o): {gpt_bias_verdict}
- LLaMA lineage (LLaMA-2 → LLaMA-3.1): {llama_bias_verdict}

**Interpretation**: RLHF-aligned models (GPT-4o, Claude Sonnet 4-6) are expected to
produce more consistent responses across demographic profiles compared to their
predecessors, but may also show reduced within-group variance (lower Fleiss' Kappa).

---

## Metrics Where 2026 Models Are Expected to Improve Over 2023

Based on prior literature and RLHF alignment properties:

1. **BERTScore F1** — larger models capture semantic similarity better
2. **METEOR** — larger vocabulary and better paraphrase handling
3. **QA Relevance** — better instruction-following
4. **Fluency (↓ PPL)** — GPT-4o and Claude are more fluent
5. **Formality (MTLD)** — RLHF models have more controlled output style
6. **Correctness (NER)** — better factual grounding

## Metrics Where 2026 Models May NOT Improve

1. **Fleiss' Kappa (alignment to human)** — RLHF training reduces variance;
   models may converge on "safe" answers regardless of persona → lower human-like diversity
2. **Chi-square p-value** — lower chi2 means more similar distributions;
   GPT-4o may actually be MORE different from 2023 human responses
3. **Demographic sensitivity (t-test)** — RLHF reduces bias but also reduces
   sensitivity to profile prompting; models may ignore demographic cues entirely

---

## Files

| File | Description |
|------|-------------|
| `results/table1_automatic_metrics.csv` | BLEU/ROUGE/METEOR/BERTScore per model |
| `results/table2_stylistic_metrics.csv` | Stylistic + GPT-4o judge scores |
| `results/table3_statistical_summary.csv` | Fleiss' Kappa, Chi-sq, t-test, ANOVA |
| `results/table4_longitudinal_comparison.csv` | Side-by-side 2023→2026 deltas |
| `results/baseline_2023.json` | Verified 2023 paper baselines |
| `results/statistical_analysis_2026.json` | Full 2026 statistical results |
| `results/qasper_automatic_metrics.csv` | Per-file Qasper automatic scores |
| `results/qasper_stylistic_metrics.csv` | Per-file Qasper stylistic scores |
| `figures/*.pdf` / `figures/*.png` | Manuscript-ready figures |

> ⏳ Items marked PENDING require running `python -m inference.run_questionnaire`
> and `python -m inference.run_qasper` first.
"""
    return md


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 62)
    print("TASK 10 — COMPILE RESULTS TABLES")
    print("=" * 62)

    print("\nBuilding Table 1 (automatic metrics) …")
    t1 = build_table1()
    out1 = RESULTS / "table1_automatic_metrics.csv"
    t1.to_csv(out1, index=False)
    print(f"  ✓ {out1.name}  ({len(t1)} rows × {len(t1.columns)} cols)")

    print("Building Table 2 (stylistic metrics) …")
    t2 = build_table2()
    out2 = RESULTS / "table2_stylistic_metrics.csv"
    t2.to_csv(out2, index=False)
    print(f"  ✓ {out2.name}  ({len(t2)} rows × {len(t2.columns)} cols)")

    print("Building Table 3 (statistical summary) …")
    t3 = build_table3()
    out3 = RESULTS / "table3_statistical_summary.csv"
    t3.to_csv(out3, index=False)
    print(f"  ✓ {out3.name}  ({len(t3)} rows × {len(t3.columns)} cols)")

    print("Building Table 4 (longitudinal comparison) …")
    t4 = build_table4(t1, t2, t3)
    out4 = RESULTS / "table4_longitudinal_comparison.csv"
    t4.to_csv(out4, index=False)
    print(f"  ✓ {out4.name}  ({len(t4)} rows × {len(t4.columns)} cols)")

    print("Building summary.md …")
    md = build_summary(t1, t2, t3, t4)
    out_md = RESULTS / "summary.md"
    with open(out_md, "w") as f:
        f.write(md)
    print(f"  ✓ {out_md.name}")

    print("\n" + "=" * 62)
    print("Tables complete. Run inference scripts to fill PENDING cells.")
    print("=" * 62)


if __name__ == "__main__":
    main()
