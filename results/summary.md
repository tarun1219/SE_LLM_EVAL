# SE_LLM_EVAL — Results Summary

**Paper:** The Kappa Paradox: RLHF-Aligned LLMs Improve Distributional Similarity
but Collapse Demographic Response Diversity in SE Survey Simulation
*(Journal of Systems & Software, manuscript under review)*

**Archived-output statement:** All analyses operate on pre-collected model
outputs in `LLM_Responses/`, `Qasper_analysis/responses/`, and `Datasets/`.
No model API calls are needed to reproduce any table or figure.

---

## Research Question Answers

### RQ1: Do 2026 LLMs simulate human survey respondents more accurately?

**Answer: No — worse. The Kappa Paradox.**

| Model | Year | Fleiss' κ | Δ vs Human Baseline |
|---|---|---|---|
| Human (target) | — | 0.1027 | 0.000 |
| GPT-3.5-Turbo | 2023 | 0.2413 | +0.139 |
| LLaMA-2-7B | 2023 | 0.2380 | +0.135 |
| GPT-4o | 2026 | 0.4804 | +0.378 ❌ further |
| Claude Sonnet 4-6 | 2026 | 0.4865 | +0.384 ❌ further |
| LLaMA-3.3-70B | 2026 | 0.4799 | +0.377 ❌ further |
| GPT-4o (P5 CoT) | 2026 | 0.4137 | +0.311 ❌ still far |
| Claude 4-6 (P5 CoT) | 2026 | 0.4137 | +0.311 ❌ still far |
| LLaMA-3.3 (P5 CoT) | 2026 | 0.4137 | +0.311 ❌ still far |

**Chi-square distributional similarity (lower χ² / higher p = more similar to human):**

| Model | χ² | p-value | Similar? |
|---|---|---|---|
| LLaMA-2-7B | 18.749 | 0.001 | ❌ No |
| GPT-4o | 2.865 | 0.581 | ✅ Yes |
| Claude 4-6 | **1.876** | **0.759** | ✅ Yes (best) |
| LLaMA-3.3-70B | 5.291 | 0.259 | ✅ Yes |

**The Distribution Paradox:** 2026 models have higher κ (worse diversity simulation)
AND higher distributional similarity to humans (lower χ²). These are reconciled
by recognizing that alignment training optimizes toward the modal human response,
which happens to match the human distribution aggregate.

---

### RQ2: Do 2026 models produce higher-quality Qasper answers? (Sanity check)

**Answer: Yes — task-specific improvement confirms the survey degradation is not
a general capability regression.**

| Metric | GPT-3.5 | LLaMA-2 | GPT-4o | Claude 4-6 | LLaMA-3.3 |
|---|---|---|---|---|---|
| BLEU-4 | 0.187 | 0.107 | 0.206 | **0.232** | 0.249* |
| ROUGE-L | 0.367 | 0.282 | 0.298 | 0.301 | **0.395*** |
| METEOR | 0.298 | 0.223 | 0.449 | **0.523** | 0.507 |
| Correctness | 0.641 | 0.502 | 0.658 | **0.762** | 0.741 |

*LLaMA-3.3's extractive style inflates BLEU/ROUGE.*

⚠️ **Metric comparability note:** TF-IDF cosine (Phase 2) ≠ DeBERTa BERTScore
(Phase 1). Phase 1 relevance/fluency methods differ from Phase 2. Only
within-Phase 2 comparisons are valid for stylistic metrics.

---

### RQ3: Did RLHF alignment reduce demographic sensitivity?

**Answer: No effect in 2023; if anything, more profile-blind in 2026.**

No model in either generation shows significant response differences across
Age, Gender, or Experience. The only significant demographic effect belongs to
human respondents (Experience: t=6.92, p<0.001).

| Model | Age p | Gender p | Exp p | Any significant? |
|---|---|---|---|---|
| Human | 0.831 | 0.722 | <0.001 ✓ | Yes (Exp) |
| GPT-3.5 | 0.242 | 0.119 | 0.341 | No |
| LLaMA-2 | 0.330 | 1.000 | 0.665 | No |
| GPT-4o | 0.964 | 0.519 | 0.870 | No |
| Claude 4-6 | 0.416 | 0.403 | 0.979 | No |
| LLaMA-3.3 | 1.000 | 0.551 | 0.689 | No |

**Important:** Absent demographic t-tests ≠ unbiased simulation. They equal
profile blindness. Both outcomes are statistically indistinguishable by t-test
alone.

---

## Practical Guidance

Before using an LLM as a synthetic respondent pool:

1. ✅ **Fleiss' κ should be within ±0.05 of your human diversity target**
   (κ > 0.35 with a human baseline near 0.10 is a red flag)
2. ✅ **Run chi-square tests** — non-significant p > 0.05 is necessary but
   not sufficient (Distribution Paradox)
3. ⚠️ **Absent demographic t-tests = profile blindness**, not accurate simulation
4. 💡 **Use P1 (Expert NLP) or P4 (Novice)** — consistently best Qasper performance
5. ❌ **CoT (P5) does not recover demographic diversity** — all three models
   converge to κ = 0.4137 with option_ids-only κ ≈ 0.007

---

## Reproduce All Results

```bash
python -m inference.run_statistics   # Fleiss κ, χ², t-tests, ANOVA
python run_metrics_pure.py           # Qasper BLEU/ROUGE/METEOR/stylistic
python -m inference.compile_tables  # CSV → paper tables
python -m visualization.generate_figures  # 5 figures
```

## Artifact Files

| File | Description |
|---|---|
| `LLM_Responses/{model}_p{n}_{shot}_responses.csv` | All archived model responses |
| `Qasper_analysis/responses/` | Qasper inference outputs |
| `results/statistical_analysis_2026.json` | Full statistical results |
| `results/qasper_automatic_metrics.csv` | Qasper automatic scores |
| `results/qasper_stylistic_metrics.csv` | Qasper stylistic scores |
| `paper/paper_jss.pdf` | Full manuscript (JSS submission draft) |
| `paper/paper_msr2027.pdf` | Earlier conference-style draft |
