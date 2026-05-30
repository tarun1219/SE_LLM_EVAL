# SE_LLM_EVAL — 2026 Longitudinal Extension

This README covers the **2026 extension** of the 2023 paper:

> *"Assessing the Effectiveness of Large Language Models as Polling Participants
> in Qualitative Research"*

The extension adds three new models (GPT-4o, Claude Sonnet 4-6, LLaMA-3.3-70B),
a new Chain-of-Thought prompt variant, GPT-4o-as-judge evaluation, and a
longitudinal comparison answering whether RLHF-aligned models simulate human survey
respondents better than their predecessors.

See [`README.md`](README.md) for the original 2023 study description.

---

## Quick-start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Set API keys
cp .env.example .env
# Edit .env with your keys (see API Keys section below)

# 3. Verify 2023 baselines reproduce correctly
python -m inference.verify_baseline

# 4. Run new model inference (questionnaire)
python -m inference.run_questionnaire --models gpt-4o claude-sonnet-4-6 llama-3.1-70b

# 5. Run Qasper inference
python -m inference.run_qasper --models gpt-4o claude-sonnet-4-6 llama-3.1-70b

# 6. Compute Qasper metrics
python -m inference.compute_qasper_metrics          # full (includes GPT-4o judge)
python -m inference.compute_qasper_metrics --no-judge  # skip judge, faster + cheaper

# 7. Statistical analysis
python -m inference.run_statistics

# 8. Compile results tables and summary
python -m inference.compile_tables

# 9. Generate figures
python -m visualization.generate_figures
```

---

## Repository Layout

```
SE_LLM_EVAL/
│
├── Datasets/                       # 2023 baselines (unchanged)
│   ├── survey_responses.csv        # Human survey (n=314)
│   ├── gpt3.5_responses.csv        # 2023 GPT-3.5 demographic responses
│   └── llama2_responses.csv        # 2023 LLaMA-2 demographic responses
│
├── LLM_Responses/                  # Profile-level response CSVs + notebooks
│   ├── gpt3.5_profile_responses.csv   # 2023 (used for chi-square)
│   ├── llama2_profile_responses.csv   # 2023
│   ├── GPT4o_responses.ipynb          # NEW 2026 — interactive GPT-4o
│   ├── Claude_responses.ipynb         # NEW 2026 — interactive Claude Sonnet 4-6
│   └── LLaMA31_responses.ipynb        # NEW 2026 — interactive LLaMA-3.3-70B
│
├── Qasper_analysis/
│   ├── Data/                       # qasper-test-v0.3.json (original)
│   ├── Metrics/                    # Original metric implementations
│   │   ├── Automatic_Metrics.py
│   │   ├── Stylistic_Metrics.py    # FIXED: model upgraded to all-mpnet-base-v2
│   │   └── Linguistic_Metrics.py
│   └── responses/                  # NEW — 2026 Qasper inference outputs
│       └── {model}_p{n}_{shot}.csv
│
├── Statistical_Analysis/
│   ├── Anova.py                    # FIXED: CSV filename, added 2026 comparison
│   └── Chi_square.py               # FIXED: removed broken utils import
│
├── LLM_Prompts/
│   └── ECS260_analysis_fleiss_kappa.ipynb  # FIXED: relative paths, kappa function
│
├── inference/                      # NEW — full 2026 pipeline
│   ├── __init__.py
│   ├── llm_client.py               # Unified OpenAI / Anthropic / Groq client
│   ├── verify_baseline.py          # Reproduces 2023 paper numbers
│   ├── run_questionnaire.py        # Questionnaire inference (Tasks 1–4)
│   ├── run_qasper.py               # Qasper inference (Task 5)
│   ├── compute_qasper_metrics.py   # Automatic + stylistic metrics (Tasks 6–7)
│   ├── run_statistics.py           # All statistical tests (Task 8)
│   └── compile_tables.py           # Results tables + summary.md (Task 10)
│
├── visualization/
│   └── generate_figures.py         # NEW — 5 manuscript-ready figures (Task 9)
│
├── results/                        # NEW — all output artefacts
│   ├── baseline_2023.json          # Verified 2023 baselines
│   ├── statistical_analysis_2026.json
│   ├── qasper_automatic_metrics.csv
│   ├── qasper_stylistic_metrics.csv
│   ├── table1_automatic_metrics.csv
│   ├── table2_stylistic_metrics.csv
│   ├── table3_statistical_summary.csv
│   ├── table4_longitudinal_comparison.csv
│   └── summary.md                  # RQ answers + improvement checklist
│
├── figures/                        # NEW — PDF + PNG at 300 DPI
│   ├── fig1_pca.{pdf,png}
│   ├── fig2_kappa.{pdf,png}
│   ├── fig3_bias_heatmap.{pdf,png}
│   ├── fig4_bertscore.{pdf,png}
│   ├── fig5_longitudinal.{pdf,png}
│   └── captions.txt
│
├── requirements.txt                # NEW — pinned deps for reproducibility
├── .env.example                    # NEW — API key template
└── README_2026.md                  # This file
```

---

## New vs Original Files

| File / Module | Status | Notes |
|---|---|---|
| `inference/` | **NEW** | Full 2026 inference + analysis pipeline |
| `visualization/generate_figures.py` | **NEW** | 5 publication-ready figures |
| `results/` | **NEW** | All output CSVs and JSONs |
| `figures/` | **NEW** | All output figures (PDF + PNG) |
| `LLM_Responses/GPT4o_responses.ipynb` | **NEW** | Interactive GPT-4o notebook |
| `LLM_Responses/Claude_responses.ipynb` | **NEW** | Interactive Claude notebook |
| `LLM_Responses/LLaMA31_responses.ipynb` | **NEW** | Interactive LLaMA-3.1 notebook |
| `requirements.txt` | **NEW** | Pinned versions for all deps |
| `.env.example` | **NEW** | API key template |
| `Statistical_Analysis/Anova.py` | **FIXED** | Wrong CSV name; added 2026 path |
| `Statistical_Analysis/Chi_square.py` | **FIXED** | Removed non-existent `utils` import |
| `Qasper_analysis/Metrics/Stylistic_Metrics.py` | **FIXED** | Upgraded sentence-transformer model |
| `LLM_Prompts/ECS260_analysis_fleiss_kappa.ipynb` | **FIXED** | Relative paths; `np.fromiter` fix |
| `LLM_Responses/GPT3.5_Responses.ipynb` | **FIXED** | OpenAI SDK v1+; removed hardcoded key |
| `Qasper_analysis/gtp_analysis.ipynb` | **FIXED** | OpenAI SDK v1+; cross-platform path |

---

## API Keys

Copy `.env.example` to `.env` and fill in your keys:

```
OPENAI_API_KEY=sk-...         # platform.openai.com/api-keys
ANTHROPIC_API_KEY=sk-ant-...  # console.anthropic.com/keys
GROQ_API_KEY=gsk_...          # console.groq.com/keys
```

> ⚠️ **Security note**: The original `Qasper_analysis/gtp_analysis.ipynb` contained
> a hardcoded OpenAI API key (`sk-6SpBf1...`). That key has been removed and replaced
> with `load_dotenv()` / `OpenAI()`. Rotate the old key at
> [platform.openai.com/api-keys](https://platform.openai.com/api-keys).

---

## Estimated API Costs

| Step | Model | Calls | Est. cost (USD) |
|------|-------|-------|-----------------|
| Questionnaire (10 profiles × 12 Q × 5 prompts × 2 shots) | GPT-4o | 1,200 | ~$3.60 |
| Questionnaire | Claude Sonnet 4-6 | 1,200 | ~$2.40 |
| Questionnaire | LLaMA-3.3-70B (Groq) | 1,200 | free tier |
| Qasper inference (50 papers × 4 prompts × 1 shot) | GPT-4o | ~800 | ~$5.00 |
| Qasper inference | Claude Sonnet 4-6 | ~800 | ~$2.00 |
| Qasper inference | LLaMA-3.3-70B (Groq) | ~800 | free tier |
| GPT-4o-as-judge (100 pairs) | GPT-4o | 100 | ~$0.40 |
| **Total (excluding Groq)** | | | **~$13.40** |

Use `--no-judge` with `compute_qasper_metrics` to skip the $0.40 GPT-4o judge cost.

---

## Prompt Variants

### Questionnaire (12 questions × demographic profiles)

| ID | Name | Shot |
|----|------|------|
| P1 | Expert NLP persona | zero / few |
| P2 | Plain QA | zero / few |
| P3 | Concise instruction | zero / few |
| P4 | Novice/child persona *(best in 2023 paper)* | zero / few |
| P5 | Chain-of-Thought *(NEW 2026)* | zero |

### Qasper (50-paper QA)

| ID | Description |
|----|-------------|
| P1 | Expert NLP extractive |
| P2 | Plain QA |
| P3 | Concise 1-sentence |
| P4 | Novice/child *(best in 2023 paper)* |

---

## 2023 Baseline Verification

Run `python -m inference.verify_baseline` to confirm the pipeline reproduces the
original paper values:

| Metric | Paper | Reproduced | Match |
|--------|-------|------------|-------|
| Human Fleiss' Kappa | 0.103 | 0.10272 | ✅ |
| GPT-3.5 Fleiss' Kappa | 0.241 | 0.24127 | ✅ |
| LLaMA-2 Fleiss' Kappa | 0.180 | 0.238 | ⚠️ |
| Chi-square (GPT-3.5 vs Human) | 18.75 | 18.7489 | ✅ |
| Chi-square p-value | 0.00088 | 0.000880 | ✅ |

> ⚠️ The LLaMA-2 kappa discrepancy (0.238 vs paper's 0.180) is because the original
> paper used a `llm_finals.csv` Colab intermediate file that was never committed to
> the repository. Human and GPT-3.5 reproduce exactly.

---

## Statistical Tests

All 5 tests from the original paper are implemented in `inference/run_statistics.py`:

1. **Fleiss' Kappa** — inter-rater agreement per respondent type
2. **t-test (demographic)** — response differences by Age / Gender / Experience
3. **Chi-square** — response distribution similarity (model vs human)
4. **Cramér's V** — effect size for chi-square
5. **One-way ANOVA** — cross-group differences on 3 key questions

Plus new longitudinal comparisons:
- GPT-3.5 → GPT-4o kappa delta and bias delta
- LLaMA-2 → LLaMA-3.1 kappa delta and bias delta

---

## Figures

Generated by `python -m visualization.generate_figures`:

| Figure | File | Description |
|--------|------|-------------|
| Fig. 1 | `fig1_pca.pdf` | PCA scatter: response clusters 2023 vs 2026 |
| Fig. 2 | `fig2_kappa.pdf` | Fleiss' Kappa bar chart with human baseline |
| Fig. 3 | `fig3_bias_heatmap.pdf` | Demographic t-test p-value heatmap (−log10) |
| Fig. 4 | `fig4_bertscore.pdf` | BERTScore F1 by model × prompt variant |
| Fig. 5 | `fig5_longitudinal.pdf` | 2023→2026 lineage improvement chart |

All figures degrade gracefully (placeholder text) when inference results are missing.

---

## Research Questions

**RQ1**: Do 2026 LLMs simulate human survey respondents more accurately?
→ Measured by Fleiss' Kappa proximity to human baseline (0.103).

**RQ2**: Do 2026 models produce higher-quality Qasper answers?
→ Measured by BLEU/ROUGE/METEOR/BERTScore (automatic) and relevance/fluency/
  formality/readability/correctness (stylistic).

**RQ3**: Did RLHF alignment reduce demographic bias in LLM responses?
→ Measured by t-test significance across Age/Gender/Experience demographics.
  Hypothesis: RLHF-aligned GPT-4o and Claude should show fewer significant
  demographic differences (less bias), but may also show lower kappa (less diversity).

See [`results/summary.md`](results/summary.md) for results once inference is complete.

---

## Citation

If you use this work, please cite the original paper and this extension:

```
@inproceedings{original2023,
  title     = {Assessing the Effectiveness of Large Language Models as Polling
               Participants in Qualitative Research},
  year      = {2023},
  ...
}
```

*(2026 extension citation TBD pending publication)*
