# SE_LLM_EVAL — 2026 Longitudinal Results Summary

Generated: 2026-05-29T12:10:16.262394

---

## Research Questions

### RQ1: Do 2026 LLMs simulate human survey respondents more accurately than 2023 models?

**Fleiss' Kappa (human baseline = 0.10272)**

| Model              | Year | Kappa  | Δ vs predecessor |
|--------------------|------|--------|-----------------|
| Human              | 2023 | 0.10272 | — |
| GPT-3.5-Turbo      | 2023 | 0.24127 | — |
| LLaMA-2-7B         | 2023 | 0.23800 | — |
| GPT-4o             | 2026 | ⏳ PENDING | ⏳ PENDING — run inference first |
| Claude Sonnet 4-6  | 2026 | ⏳ PENDING | — |
| LLaMA-3.1-70B      | 2026 | ⏳ PENDING | ⏳ PENDING — run inference first |

---

### RQ2: Do 2026 models produce higher-quality Qasper answers?

**GPT lineage (GPT-3.5-Turbo → GPT-4o):**
  - BLEU-4: ⏳ PENDING
  - ROUGE-L: ⏳ PENDING
  - METEOR: ⏳ PENDING
  - BERTScore F1: ⏳ PENDING
  - QA Relevance: ⏳ PENDING
  - Fluency (PPL): ⏳ PENDING

**LLaMA lineage (LLaMA-2-7B → LLaMA-3.1-70B):**
  - BLEU-4: ⏳ PENDING
  - ROUGE-L: ⏳ PENDING
  - METEOR: ⏳ PENDING
  - BERTScore F1: ⏳ PENDING
  - QA Relevance: ⏳ PENDING
  - Fluency (PPL): ⏳ PENDING

---

### RQ3: Did RLHF alignment reduce demographic bias in LLM responses?

The t-tests measure whether LLM responses differ significantly across demographic profiles
(Age, Gender, Experience). Fewer significant differences = less demographic bias.

- GPT lineage (GPT-3.5 → GPT-4o): ⏳ PENDING
- LLaMA lineage (LLaMA-2 → LLaMA-3.1): ⏳ PENDING

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
