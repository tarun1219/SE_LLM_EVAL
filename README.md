# SE_LLM_EVAL — LLMs as Synthetic Survey Respondents

> **A longitudinal study evaluating whether Large Language Models can simulate human survey participants in software engineering qualitative research.**

This repository contains the full code, data, and paper for a **two-generation comparison** (2023 → 2026) of LLMs acting as synthetic respondents to a 12-question software developer survey, plus a Qasper scientific QA evaluation.

---

## What This Study Asks

Can LLMs stand in for human participants in qualitative SE research? Specifically:

| Research Question | Short Answer |
|---|---|
| **RQ1** — Do 2026 models simulate human survey responses better than 2023 models? | ❌ Worse. RLHF collapses response variance (Kappa Paradox). |
| **RQ2** — Do 2026 models produce better Qasper scientific QA answers? | ✅ Yes. METEOR improves by up to 75%. |
| **RQ3** — Did RLHF reduce demographic sensitivity? | ⚠️ Already absent in 2023. 2026 models show even stronger profile-blindness. |

---

## Key Finding: The Kappa Paradox

2026 RLHF-aligned models achieve **~2× higher Fleiss' Kappa** than 2023 models — but this is *bad*. Higher kappa means they give the same answer regardless of who they're supposed to be simulating.

| Model | Year | Fleiss' κ | Δ vs Human Baseline |
|---|---|---|---|
| Human respondents | 2023 | 0.103 | — (target) |
| GPT-3.5-Turbo | 2023 | 0.241 | +0.139 |
| LLaMA-2-7B | 2023 | 0.238 | +0.135 |
| **GPT-4o** | **2026** | **0.480** | **+0.378** |
| **Claude Sonnet 4-6** | **2026** | **0.487** | **+0.384** |
| **LLaMA-3.3-70B** | **2026** | **0.480** | **+0.377** |

All three 2026 models from *three different organizations* converge to κ ∈ [0.480, 0.487] — confirming RLHF itself (not architecture) drives response uniformity.

---

## Repository Layout

```
SE_LLM_EVAL/
│
├── paper/                          ← PAPER OUTPUT (new 2026)
│   ├── paper.tex                   ← Full LaTeX source
│   └── paper.pdf                   ← Compiled PDF (all results, 7 tables)
│
├── inference/                      ← Pipeline code (new 2026)
│   ├── llm_client.py               ← Unified OpenAI / Anthropic / Groq client
│   ├── run_questionnaire.py        ← Survey inference (GPT-4o, Claude, LLaMA)
│   ├── run_qasper.py               ← Qasper QA inference
│   ├── run_statistics.py           ← Fleiss κ, chi-square, t-test, ANOVA
│   └── compile_tables.py           ← Generates results tables + summary
│
├── visualization/
│   └── generate_figures.py         ← 5 publication-ready figures (PDF + PNG)
│
├── results/                        ← All computed outputs (new 2026)
│   ├── statistical_analysis_2026.json
│   ├── qasper_automatic_metrics.csv
│   ├── qasper_stylistic_metrics.csv
│   ├── table1_automatic_metrics.csv
│   ├── table2_stylistic_metrics.csv
│   ├── table3_statistical_summary.csv
│   ├── table4_longitudinal_comparison.csv
│   ├── paper_draft.md              ← Markdown version of the paper
│   └── summary.md                  ← RQ answers + key numbers at a glance
│
├── figures/                        ← Generated figures (new 2026)
│   ├── fig1_pca_comparison.{pdf,png}
│   ├── fig2_fleiss_kappa_longitudinal.{pdf,png}
│   ├── fig3_demographic_bias_heatmap.{pdf,png}
│   ├── fig4_bertscore_comparison.{pdf,png}
│   ├── fig5_longitudinal_improvement.{pdf,png}
│   └── captions.txt
│
├── Datasets/                       ← 2023 baselines (unchanged)
│   ├── survey_responses.csv        ← Human survey (n=314)
│   ├── gpt3.5_responses.csv        ← 2023 GPT-3.5 responses
│   └── llama2_responses.csv        ← 2023 LLaMA-2 responses
│
├── LLM_Responses/                  ← Per-profile response CSVs
│   └── {model}_p{n}_{shot}_responses.csv
│
├── Qasper_analysis/
│   ├── Data/                       ← qasper-test-v0.3.json
│   └── responses/                  ← 2026 Qasper inference outputs
│
├── Statistical_Analysis/           ← Original 2023 stat scripts (fixed)
├── run_metrics_pure.py             ← Zero-ML metrics pipeline (no GPU needed)
├── requirements.txt                ← Pinned dependencies
└── README_2026.md                  ← Detailed 2026 extension docs
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # optional, for NER metrics
```

### 2. Set API Keys

```bash
cp .env.example .env
# Edit .env with your keys:
#   OPENAI_API_KEY=sk-...
#   ANTHROPIC_API_KEY=sk-ant-...
#   GROQ_API_KEY=gsk_...
```

### 3. Run the Full Pipeline

```bash
# Questionnaire inference (10 profiles × 12 questions × 5 prompts)
python -m inference.run_questionnaire --models gpt-4o claude-sonnet-4-6 llama-3.3-70b

# Qasper QA inference (50 papers, 4 prompts, few-shot)
python -m inference.run_qasper --models gpt-4o claude-sonnet-4-6 llama-3.3-70b

# Compute Qasper metrics (pure Python, no GPU needed)
python run_metrics_pure.py

# Statistical analysis (kappa, chi-square, t-tests, ANOVA)
python -m inference.run_statistics

# Compile result tables
python -m inference.compile_tables

# Generate all 5 figures
python -m visualization.generate_figures
```

### 4. View the Paper

Open `paper/paper.pdf` — it contains all results, tables, and discussion.

---

## Models Compared

| Model | Year | Provider | Key Difference from Predecessor |
|---|---|---|---|
| GPT-3.5-Turbo | 2023 | OpenAI | Baseline |
| LLaMA-2-7B | 2023 | Meta | Baseline (7B params) |
| GPT-4o | 2026 | OpenAI | Multimodal, enhanced RLHF |
| Claude Sonnet 4-6 | 2026 | Anthropic | Constitutional AI (RLAIF) |
| LLaMA-3.3-70B | 2026 | Meta | 10× scale increase, stronger RLHF |

---

## Results at a Glance

### Survey Simulation (Questionnaire)

| Metric | 2023 Best | 2026 Best | Direction |
|---|---|---|---|
| Fleiss' Kappa (closer to 0.103 = better) | 0.241 (GPT-3.5) | 0.480 (GPT-4o) | ❌ Worse |
| Chi-square vs reference (lower = better) | 18.75 | 1.876 (Claude) | ✅ Better |
| Demographic t-test significance | 0/3 dimensions | 0/3 dimensions | ➡️ Same (profile-blind) |

### Scientific QA (Qasper, 124 pairs)

| Metric | GPT-3.5 (2023) | LLaMA-2 (2023) | Claude (2026) | LLaMA-3.3 (2026) |
|---|---|---|---|---|
| BLEU-4 | 0.187 | 0.107 | **0.232** | **0.249** |
| ROUGE-L | 0.367 | 0.282 | 0.301 | **0.395** |
| METEOR | 0.298 | 0.223 | **0.523** | 0.507 |
| Correctness | 0.641 | 0.502 | **0.762** | 0.741 |

---

## Practical Advice for SE Researchers

Before using an LLM as a synthetic respondent pool:

1. ✅ **Check Fleiss' Kappa** — it should be within ±0.05 of your expected human diversity
2. ✅ **Run chi-square tests** — non-significant results (p > 0.05) indicate distributional similarity
3. ⚠️ **Run demographic t-tests** — absence of significant differences is a signal of *profile-blindness*, not unbiased simulation
4. 💡 **Use P1 (Expert NLP) or P4 (Novice Persona)** prompts — they consistently outperform simpler variants

---

## Prompt Variants

| ID | Name | Description |
|---|---|---|
| P1 | Expert NLP | Model framed as NLP expert; strict extractive answers |
| P2 | Plain QA | Minimal framing; generic QA |
| P3 | Concise | One-sentence answers only |
| P4 | Novice/Child | Model plays a novice with no domain knowledge *(best 2023 alignment)* |
| P5 | Chain-of-Thought | Step-by-step reasoning grounded in demographic profile *(NEW 2026)* |

---

## Reproducing 2023 Baselines

```bash
python -m inference.verify_baseline
```

| Metric | Paper | Reproduced | Status |
|---|---|---|---|
| Human Fleiss' κ | 0.103 | 0.10272 | ✅ |
| GPT-3.5 Fleiss' κ | 0.241 | 0.24127 | ✅ |
| Chi-square (GPT-3.5) | 18.75 | 18.749 | ✅ |
| Chi-square p-value | 0.00088 | 0.000880 | ✅ |
| LLaMA-2 Fleiss' κ | 0.180 | 0.238 | ⚠️ |

> ⚠️ The LLaMA-2 kappa discrepancy (0.238 vs 0.180) is because the original study
> used an intermediate Colab file (`llm_finals.csv`) that was never committed to the repo.
> Human and GPT-3.5 results reproduce exactly.

---

## Estimated API Costs

| Step | Model | Calls | Est. Cost |
|---|---|---|---|
| Questionnaire (1200 calls) | GPT-4o | 1,200 | ~$3.60 |
| Questionnaire | Claude Sonnet 4-6 | 1,200 | ~$2.40 |
| Questionnaire | LLaMA-3.3-70B (Groq) | 1,200 | free tier |
| Qasper inference | GPT-4o | ~800 | ~$5.00 |
| Qasper inference | Claude Sonnet 4-6 | ~800 | ~$2.00 |
| Qasper inference | LLaMA-3.3-70B | ~800 | free tier |
| **Total** | | | **~$13** |

---

## Read the Paper

The full paper is available in two formats:

- **PDF**: [`paper/paper.pdf`](paper/paper.pdf) — compiled LaTeX, all tables and figures
- **Markdown**: [`results/paper_draft.md`](results/paper_draft.md) — quick reading

The paper covers:
- Abstract and Introduction
- Related Work (silicon sampling, RLHF diversity, prompt engineering, Qasper, metrics)
- Methodology (profiles, prompts, models, statistical tests)
- Results (7 tables, 5 figures, all RQ answers)
- Discussion (Kappa Paradox, distribution paradox, profile blindness)
- Limitations and Conclusion
- References (11 entries including SSRN working paper)

---

## Citation

If you use this work, please cite:

```bibtex
@misc{sellmeval2026,
  title     = {Assessing the Effectiveness of Large Language Models as Polling
               Participants in Qualitative Research: A 2023--2026 Longitudinal Extension},
  author    = {Gogineni, Tarun},
  year      = {2026},
  note      = {Repository: SE\_LLM\_EVAL},
  url       = {https://github.com/tarungogineni/SE_LLM_EVAL}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
