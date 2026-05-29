"""
visualization/generate_figures.py
Task 9 — Generate all 5 paper figures.

Figure 1 : PCA scatter — 2023 vs 2026 side-by-side subplots
Figure 2 : Fleiss' Kappa bar chart (longitudinal, 2023 light / 2026 dark)
Figure 3 : Demographic bias heatmap (p-values, red = significant)
Figure 4 : BERTScore F1 grouped bar chart by prompt type
Figure 5 : Longitudinal improvement chart (per-metric delta)

Saves PDF (300 DPI) + PNG to figures/.
Also writes figures/captions.txt.

Run:
    cd /path/to/SE_LLM_EVAL
    python -m visualization.generate_figures
"""

from __future__ import annotations
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

REPO    = Path(__file__).parent.parent
RES     = REPO / "results"
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Style constants ───────────────────────────────────────────────────────────
COLORS = {
    "human":  "#E53935",   # red
    "gpt35":  "#1565C0",   # dark blue  (2023)
    "llama2": "#2E7D32",   # dark green (2023)
    "gpt4o":  "#42A5F5",   # light blue (2026)
    "claude": "#AB47BC",   # purple     (2026)
    "llama31":"#66BB6A",   # light green(2026)
}
DISPLAY = {
    "human":  "Human", "gpt35": "GPT-3.5",
    "llama2": "LLaMA-2", "gpt4o": "GPT-4o",
    "claude": "Claude S", "llama31": "LLaMA-3.1",
}
YEAR_ALPHA = {"2023": 0.55, "2026": 0.90}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def _save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {name}.pdf + .png")


# ── Figure 1: PCA ─────────────────────────────────────────────────────────────

def fig1_pca() -> str:
    """PCA of response distributions, 2023 vs 2026."""
    DATA = REPO / "Datasets"
    LLM  = REPO / "LLM_Responses"

    # --- build a unified dataframe with respondent_type and response_encoded ---
    frames = []
    spec = [
        ("human",   DATA / "survey_responses.csv",        "Option",  "2023"),
        ("gpt35",   DATA / "gpt3.5_responses.csv",        "Option",  "2023"),
        ("llama2",  DATA / "llama2_responses.csv",         "Answer",  "2023"),
    ]
    for slug, path, col, year in spec:
        if not path.exists():
            continue
        df = pd.read_csv(path)[[col, "Question"]].rename(columns={col: "ans"})
        df["respondent_type"] = slug
        df["year"] = year
        frames.append(df)

    for slug, fname, year in [
        ("gpt4o",   "gpt4o_p4_few_responses.csv",     "2026"),
        ("claude",  "claudesonnet46_p4_few_responses.csv", "2026"),
        ("llama31", "llama3170b_p4_few_responses.csv", "2026"),
    ]:
        p = LLM / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        col = "Option IDs" if "Option IDs" in df.columns else "Answer"
        df = df[[col, "question" if "question" in df.columns else "Question"]].copy()
        df.columns = ["ans", "Question"]
        df["respondent_type"] = slug
        df["year"] = year
        frames.append(df)

    if not frames:
        print("  [fig1] No data found — creating placeholder")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, t in zip(axes, ["2023 Baseline", "2026 New Models"]):
            ax.text(0.5, 0.5, f"{t}\n(run inference first)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=13, color="grey")
            ax.set_title(t)
        _save(fig, "fig1_pca_comparison")
        return "Figure 1: PCA could not be generated (no 2026 data yet)."

    all_df = pd.concat(frames, ignore_index=True)
    le = LabelEncoder()
    all_df["enc"] = le.fit_transform(all_df["ans"].astype(str))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, year, title in [(axes[0], "2023", "2023 Baseline"),
                             (axes[1], "2026", "2026 New Models")]:
        sub = all_df[all_df["year"] == year]
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            ax.set_title(title)
            continue
        pca = PCA(n_components=2, random_state=42)
        comps = pca.fit_transform(sub[["enc"]].values)
        for rtype in sub["respondent_type"].unique():
            mask = sub["respondent_type"] == rtype
            ax.scatter(comps[mask, 0], comps[mask, 1],
                       label=DISPLAY.get(rtype, rtype),
                       color=COLORS.get(rtype, "grey"),
                       alpha=0.65, s=40, edgecolors="white", linewidths=0.5)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.legend(framealpha=0.9, fontsize=9)

    fig.suptitle("PCA of Response Distributions: 2023 vs 2026", fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "fig1_pca_comparison")
    return (
        "Figure 1: PCA scatter plots of response distributions for 2023 baseline models "
        "(GPT-3.5, LLaMA-2, Human) and 2026 new models (GPT-4o, Claude Sonnet, LLaMA-3.1). "
        "Each point represents one survey response encoded as a numeric category. "
        "PCA reduces the high-dimensional option space to two components."
    )


# ── Figure 2: Fleiss' Kappa bar chart ────────────────────────────────────────

def fig2_kappa() -> str:
    stats_path = RES / "statistical_analysis_2026.json"
    if not stats_path.exists():
        print("  [fig2] statistical_analysis_2026.json missing — using baseline only")
        base_path = RES / "baseline_2023.json"
        if not base_path.exists():
            print("  [fig2] baseline_2023.json also missing — placeholder")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Run inference.run_statistics first",
                    ha="center", va="center", transform=ax.transAxes, color="grey")
            _save(fig, "fig2_fleiss_kappa_longitudinal")
            return "Figure 2: Fleiss' Kappa (data not yet available)."
        with open(base_path) as f:
            data = json.load(f)
        kappa = data["fleiss_kappa"]
    else:
        with open(stats_path) as f:
            data = json.load(f)
        kappa = data["fleiss_kappa"]

    order_2023 = ["human", "gpt35", "llama2"]
    order_2026 = ["gpt4o", "claude", "llama31"]

    labels, values, colors_bar, years = [], [], [], []
    for key in order_2023:
        if key in kappa:
            labels.append(DISPLAY[key])
            values.append(kappa[key])
            colors_bar.append("#BBDEFB" if key != "human" else "#FFCDD2")
            years.append("2023")
    for key in order_2026:
        if key in kappa:
            labels.append(DISPLAY[key])
            values.append(kappa[key])
            colors_bar.append(COLORS[key])
            years.append("2026")

    human_kappa = kappa.get("human", None)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors_bar, edgecolor="white", linewidth=0.8,
                  width=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    if human_kappa is not None:
        ax.axhline(y=human_kappa, color="black", linestyle="--", linewidth=1.5,
                   label=f"Human baseline ({human_kappa:.3f})")

    patch_2023 = mpatches.Patch(facecolor="#BBDEFB", label="2023 models")
    patch_2026 = mpatches.Patch(facecolor="#42A5F5", label="2026 models")
    handles = [patch_2023, patch_2026]
    if human_kappa is not None:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color="black", linestyle="--", label="Human baseline"))
    ax.legend(handles=handles, framealpha=0.9)

    ax.set_xlabel("Respondent Type")
    ax.set_ylabel("Fleiss' Kappa")
    ax.set_title("Response Agreement (Fleiss' Kappa): 2023 vs 2026 Models",
                 fontweight="bold")
    ax.set_ylim(0, max(values) * 1.2 if values else 0.4)
    plt.tight_layout()
    _save(fig, "fig2_fleiss_kappa_longitudinal")
    return (
        "Figure 2: Bar chart of Fleiss' Kappa scores for each respondent type. "
        "Light bars = 2023 models, dark bars = 2026 models. "
        "The dashed line marks the human baseline. Higher kappa indicates more "
        "uniform (less diverse) responses."
    )


# ── Figure 3: Demographic bias heatmap ──────────────────────────────────────

def fig3_bias_heatmap() -> str:
    stats_path = RES / "statistical_analysis_2026.json"
    if not stats_path.exists():
        stats_path = RES / "baseline_2023.json"
    if not stats_path.exists():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Run inference.run_statistics first",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        _save(fig, "fig3_demographic_bias_heatmap")
        return "Figure 3: Demographic bias heatmap (data not yet available)."

    with open(stats_path) as f:
        data = json.load(f)
    ttests = data.get("ttest", data.get("t_tests", {}))

    models  = [k for k in ttests if "profile" not in k]
    demos   = ["Age", "Gender", "Experience"]
    p_grid  = np.full((len(models), len(demos)), np.nan)

    for i, model in enumerate(models):
        for j, demo in enumerate(demos):
            d = ttests.get(model, {}).get(demo, {})
            if isinstance(d, dict):
                for pair_data in d.values():
                    if isinstance(pair_data, dict) and "p" in pair_data:
                        p_grid[i, j] = pair_data["p"]
                        break

    # Convert p-values to -log10 for colour intensity; cap at 5
    log_p = np.where(np.isnan(p_grid), np.nan, np.minimum(-np.log10(np.clip(p_grid, 1e-10, 1)), 5))

    fig, ax = plt.subplots(figsize=(8, len(models) * 0.8 + 1.5))
    im = ax.imshow(log_p, cmap="Reds", aspect="auto", vmin=0, vmax=5)

    ax.set_xticks(range(len(demos)))
    ax.set_xticklabels(demos, fontsize=11)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([DISPLAY.get(m, m) for m in models], fontsize=11)

    for i in range(len(models)):
        for j in range(len(demos)):
            p_val = p_grid[i, j]
            txt = f"p={p_val:.3f}" if not np.isnan(p_val) else "n/a"
            sig = "*" if (not np.isnan(p_val) and p_val < 0.05) else ""
            ax.text(j, i, f"{txt}{sig}", ha="center", va="center",
                    fontsize=8.5, color="white" if (not np.isnan(p_val) and p_val < 0.01) else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("-log₁₀(p-value)", fontsize=10)
    cbar.ax.set_yticks([0, 1.3, 2, 5])
    cbar.ax.set_yticklabels(["p=1.0", "p=0.05*", "p=0.01**", "p=10⁻⁵"])

    ax.set_title("Demographic Bias p-values\n(red = statistically significant)",
                 fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig3_demographic_bias_heatmap")
    return (
        "Figure 3: Heatmap of t-test p-values comparing response distributions "
        "across demographic groups (age, gender, experience) for each model. "
        "Red cells (marked *) indicate statistically significant differences (p < 0.05), "
        "suggesting the model produces systematically different responses for "
        "different demographic profiles."
    )


# ── Figure 4: BERTScore F1 grouped bar chart ─────────────────────────────────

def fig4_bertscore() -> str:
    auto_path = RES / "qasper_automatic_metrics.csv"
    if not auto_path.exists():
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "Run inference.compute_qasper_metrics first",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        _save(fig, "fig4_bertscore_comparison")
        return "Figure 4: BERTScore comparison (data not yet available)."

    df = pd.read_csv(auto_path)
    prompts = sorted(df["prompt_num"].unique())
    models  = df["model"].unique().tolist()

    x = np.arange(len(prompts))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("prompt_num")
        vals = [sub.loc[p, "bertscore_f1"] if p in sub.index else 0 for p in prompts]
        slug = model.replace("-","").replace(".","").replace(":","")[:10]
        color = COLORS.get(slug, f"C{i}")
        bars = ax.bar(x + i * width, vals, width, label=DISPLAY.get(slug, model),
                      color=color, alpha=0.85)

    ax.set_xlabel("Prompt Variant")
    ax.set_ylabel("BERTScore F1")
    ax.set_title("BERTScore F1 by Model and Prompt Variant", fontweight="bold")
    ax.set_xticks(x + width * (len(models)-1) / 2)
    ax.set_xticklabels([f"P{p}" for p in prompts])
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    _save(fig, "fig4_bertscore_comparison")
    return (
        "Figure 4: Grouped bar chart of BERTScore F1 scores on the Qasper QA task "
        "for each model (grouped by prompt variant P1–P4). "
        "BERTScore uses DeBERTa embeddings to measure semantic similarity between "
        "generated answers and gold references."
    )


# ── Figure 5: Longitudinal improvement chart ─────────────────────────────────

def fig5_longitudinal() -> str:
    stats_path = RES / "statistical_analysis_2026.json"
    if not stats_path.exists():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Run inference.run_statistics first",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        _save(fig, "fig5_longitudinal_improvement")
        return "Figure 5: Longitudinal improvement (data not yet available)."

    with open(stats_path) as f:
        data = json.load(f)

    kappa = data.get("fleiss_kappa", {})
    long_ = data.get("longitudinal", {})

    metrics = ["Fleiss' Kappa", "Bias Score"]
    gpt_vals  = [
        (kappa.get("gpt35",  np.nan), kappa.get("gpt4o",  np.nan)),
        (long_.get("bias_gpt35_to_gpt4o",  {}).get("old", np.nan),
         long_.get("bias_gpt35_to_gpt4o",  {}).get("new", np.nan)),
    ]
    llama_vals = [
        (kappa.get("llama2", np.nan), kappa.get("llama31", np.nan)),
        (long_.get("bias_llama2_to_llama31", {}).get("old", np.nan),
         long_.get("bias_llama2_to_llama31", {}).get("new", np.nan)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, lineage, vals, (c23, c26) in [
        (axes[0], "GPT lineage",   gpt_vals,  ("#1565C0", "#42A5F5")),
        (axes[1], "LLaMA lineage", llama_vals,("#2E7D32", "#66BB6A")),
    ]:
        for mi, (metric, (v23, v26)) in enumerate(zip(metrics, vals)):
            if np.isnan(v23) or np.isnan(v26):
                continue
            ax.plot([0, 1], [v23, v26], "o-", color=c23 if mi == 0 else c26,
                    linewidth=2, markersize=8, label=metric)
            ax.annotate(f"{v23:.3f}", (0, v23), textcoords="offset points",
                        xytext=(-18, 4), fontsize=9)
            ax.annotate(f"{v26:.3f}", (1, v26), textcoords="offset points",
                        xytext=(4, 4), fontsize=9)

        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["2023", "2026"])
        ax.set_title(lineage, fontweight="bold")
        ax.set_ylabel("Metric value")
        ax.legend(fontsize=9)

    fig.suptitle("Longitudinal Change in Key Metrics (2023 → 2026)", fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig5_longitudinal_improvement")
    return (
        "Figure 5: Longitudinal comparison of Fleiss' Kappa and demographic bias scores "
        "for the GPT lineage (GPT-3.5 → GPT-4o) and LLaMA lineage (LLaMA-2 → LLaMA-3.1). "
        "Each line connects the 2023 value (left) to the 2026 value (right), "
        "showing directional change in response uniformity and demographic sensitivity."
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("TASK 9 — GENERATE ALL FIGURES")
    print("=" * 60)

    captions: list[str] = []
    for i, fn in enumerate([fig1_pca, fig2_kappa, fig3_bias_heatmap,
                             fig4_bertscore, fig5_longitudinal], start=1):
        print(f"\nFigure {i}:")
        caption = fn()
        captions.append(f"Figure {i}: {caption}")

    cap_path = FIG_DIR / "captions.txt"
    cap_path.write_text("\n\n".join(captions))
    print(f"\n✓ Captions saved → figures/captions.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
