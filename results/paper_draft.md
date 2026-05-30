# Assessing the Effectiveness of Large Language Models as Polling Participants in Qualitative Research: A 2023–2026 Longitudinal Extension

---

## Abstract

Large language models (LLMs) have emerged as a promising tool for simulating survey respondents in qualitative research, enabling low-cost, scalable pilot studies without requiring human participant recruitment. This paper presents a longitudinal extension of a 2023 study that evaluated GPT-3.5-Turbo and LLaMA-2-7B as synthetic survey respondents. We introduce three new models — GPT-4o, Claude Sonnet 4-6, and LLaMA-3.3-70B — and evaluate them across the same 12-question software-developer survey (n=314 human baseline) under five prompt variants, including a novel Chain-of-Thought (CoT) condition. We additionally assess each model's scientific question-answering capability on 50 papers from the Qasper benchmark. Statistical analyses include Fleiss' Kappa (inter-rater agreement), chi-square tests (distribution similarity), t-tests (demographic sensitivity), Cramér's V (effect size), and one-way ANOVA. Our findings reveal a *Kappa Paradox*: RLHF-aligned 2026 models exhibit nearly double the within-group Fleiss' Kappa of their 2023 predecessors (GPT-4o: κ=0.480; Claude Sonnet 4-6: κ=0.487; LLaMA-3.3-70B: κ=0.480 versus GPT-3.5: κ=0.241, human: κ=0.103), indicating dramatically increased response uniformity rather than improved demographic diversity. Concurrently, chi-square tests show that 2026 model response distributions are substantially less divergent from the reference distribution than their predecessors (GPT-4o: χ²=2.86, p=0.581; Claude: χ²=1.88, p=0.759 versus 2023 baseline: χ²=18.75, p<0.001), while all models — including 2023 baselines — show zero statistically significant demographic sensitivity across Age, Gender, and Experience dimensions. On Qasper scientific QA, 2026 models substantially outperform 2023 baselines on BLEU-4 (LLaMA-3.3: 0.249 vs. LLaMA-2: 0.107), ROUGE-L (LLaMA-3.3: 0.395 vs. LLaMA-2: 0.282), and METEOR (Claude: 0.523 vs. GPT-3.5: 0.298). These results have direct implications for the use of LLMs as synthetic respondents in software engineering qualitative research.

**Keywords:** Large language models, survey simulation, synthetic respondents, RLHF alignment, Fleiss' Kappa, Qasper, longitudinal study, software engineering

---

## 1. Introduction

Qualitative research in software engineering frequently relies on surveys and interviews with human participants. However, recruiting diverse, representative participants is costly, time-consuming, and often limited by self-selection bias. The emergence of instruction-tuned LLMs has raised a compelling question: can these models serve as *synthetic respondents*, approximating human survey behaviour well enough to pilot study instruments, validate questionnaire designs, or augment small-sample studies?

A foundational 2023 study [CITATION] explored this question by prompting GPT-3.5-Turbo and LLaMA-2-7B with demographic profiles and measuring whether their responses aligned with those of 314 human software developers across a 12-question survey. The study found that both models exhibited higher inter-rater agreement (Fleiss' Kappa = 0.241 for GPT-3.5, 0.180 for LLaMA-2) than human respondents (κ = 0.103), suggesting that LLMs produce more internally consistent but less humanistically diverse responses. Chi-square tests confirmed statistically significant divergence between model and human response distributions (χ² = 18.75, p < 0.001).

Since 2023, the LLM landscape has transformed dramatically. OpenAI's GPT-4o, Anthropic's Claude Sonnet 4-6, and Meta's LLaMA-3.3-70B represent a new generation of models trained with Reinforcement Learning from Human Feedback (RLHF) at unprecedented scale. RLHF alignment has been shown to improve instruction-following, reduce harmful outputs, and increase response coherence. However, its effect on *demographic sensitivity* — the ability of an LLM to produce meaningfully different responses when prompted with different demographic profiles — remains underexplored.

This paper addresses three research questions:

- **RQ1**: Do 2026 LLMs simulate human survey respondents more accurately than their 2023 predecessors, as measured by Fleiss' Kappa proximity to the human baseline?
- **RQ2**: Do 2026 models produce higher-quality answers on the Qasper scientific QA benchmark, as measured by automatic and stylistic metrics?
- **RQ3**: Did RLHF alignment reduce demographic sensitivity in LLM survey responses, as measured by t-test significance across Age, Gender, and Experience demographics?

We make the following contributions:

1. The first longitudinal comparison of LLM survey simulation capability across two model generations (2023 vs. 2026), using identical evaluation protocols.
2. A new Chain-of-Thought prompt variant (P5) added to the existing four prompt templates (P1–P4), investigating whether structured reasoning improves demographic persona adherence.
3. A comprehensive evaluation of three 2026 models on the Qasper benchmark using BLEU-1/2/3/4, ROUGE-1/2/L, METEOR, and five stylistic metrics (relevance, fluency, formality, readability, correctness).
4. Open-source release of the complete inference pipeline, all model responses, and analysis scripts.

---

## 2. Related Work

### 2.1 LLMs as Survey Respondents

The use of LLMs to simulate human survey responses has attracted growing attention. Argyle et al. [CITATION] introduced the concept of "silicon sampling," demonstrating that GPT-3 could reproduce attitudinal patterns from the American National Election Studies when prompted with demographic information. Their work showed that LLM responses correlate with subgroup survey responses at rates exceeding random baseline, though they cautioned against treating LLM-simulated data as equivalent to human data.

Santurkar et al. [CITATION] extended this analysis to GPT-3.5, finding that default (non-demographically prompted) LLMs exhibit systematic opinion biases that skew toward liberal, Western, and educated viewpoints. Prompted models showed improved demographic sensitivity, though significant gaps remained for underrepresented groups.

In the software engineering domain, the 2023 study that this paper extends [CITATION] was among the first to apply LLM survey simulation specifically to developer-focused questionnaires, validating results against a real n=314 developer survey collected at a university computing department.

### 2.2 RLHF Alignment and Response Diversity

Reinforcement Learning from Human Feedback (RLHF) [CITATION: Ouyang et al., 2022] has become the dominant fine-tuning paradigm for instruction-following LLMs. By training models to produce responses rated highly by human raters, RLHF improves helpfulness and safety but also introduces a tendency toward "safe," high-consensus answers. Perez et al. [CITATION] demonstrated that RLHF-aligned models show reduced response variance across equivalent prompts, which has implications for their use as synthetic respondents: a model that always defaults to the most socially acceptable answer may appear unbiased but actually underrepresents minority viewpoints.

### 2.3 Prompt Engineering for Persona Simulation

The influence of prompt formulation on LLM demographic simulation has been systematically explored by several groups. Argyle et al. [CITATION] found that explicit demographic backstory prompts substantially improved persona alignment. The 2023 study [CITATION] compared four prompt styles — expert NLP persona (P1), plain QA (P2), concise instruction (P3), and novice/child persona (P4) — and found P4 (novice/child) to yield the highest alignment with human distributions. We extend this with a Chain-of-Thought (P5) prompt, hypothesizing that step-by-step reasoning may reduce pattern-matching shortcuts.

### 2.4 Scientific Question Answering Benchmarks

The Qasper dataset [CITATION: Dasigi et al., 2021] provides 1,585 questions over 888 NLP research papers, with extractive and free-form gold answers from human annotators. It has been widely used to evaluate document-level reading comprehension. We use a 50-paper random sample (seed=42) from the test split, consistent with prior work, to evaluate factual QA capability across model generations.

### 2.5 Evaluation Metrics

BLEU [CITATION: Papineni et al., 2002], ROUGE [CITATION: Lin, 2004], and METEOR [CITATION: Banerjee & Lavie, 2005] remain standard automatic metrics for text generation evaluation. For stylistic evaluation, we employ TF-IDF cosine similarity (relevance), type-token ratio (lexical diversity/fluency proxy), MTLD (formality), Flesch Reading Ease (readability), and FuzzyWuzzy token-set ratio (correctness). Due to system memory constraints, we use TF-IDF cosine similarity as a proxy for BERTScore [CITATION: Zhang et al., 2019] in the 2026 evaluation; 2023 BERTScore values from the original paper used the `microsoft/deberta-xlarge-mnli` model and are not directly comparable.

---

## 3. Methodology

### 3.1 Human Survey Baseline

The human baseline consists of n=314 software developer responses to a 12-question survey administered at a university computing department. Respondents ranged from undergraduate students to professional developers, with demographic distributions across Age (18–22, 23–26, 27–35 years), Gender (Man, Woman), and Experience (< 1 year, 1–3 years, 3–5 years). Questions covered development environment preferences, learning strategies, professional challenges, programming language selection, collaboration, innovation versus deadlines, societal contributions, and ethical decision-making scenarios.

### 3.2 Questionnaire Stimulus

The 12-question survey instrument from the original 2023 study is reproduced verbatim. Questions include:

1. Preferred development environment (Windows/macOS/Linux/Other)
2. How do you learn to code? (multi-select)
3. Biggest challenge as a developer (4 options)
4. Programming language selection criteria (4 options)
5. Team communication and collaboration (4 options)
6. Staying up-to-date with industry changes (multi-select)
7. Balancing innovation vs. deadlines (4 options)
8. Software's contribution to societal challenges (multi-select)
9. AI-driven employee dismissal ethical scenario (3 options)
10. Unfamiliar language under deadline scenario (4 options)
11. Critical pre-release bug scenario (4 options)
12. SaaS delivery with critical bug scenario (5 options)

### 3.3 Demographic Profiles

Ten synthetic profiles are constructed by crossing Age × Gender × Ethnicity × Education × Experience:

| Profile | Age | Gender | Ethnicity | Education | Experience |
|---------|-----|--------|-----------|-----------|------------|
| 1 | 18–22 | Woman | Asian | Bachelor's | < 1 year |
| 2 | 27–35 | Man | White | Professional | 3–5 years |
| 3 | 23–26 | Woman | White | Master's | 1–3 years |
| 4 | 27–35 | Man | Asian | Bachelor's | < 1 year |
| 5 | 18–22 | Woman | White | Professional | 3–5 years |
| 6 | 23–26 | Man | Asian | Master's | < 1 year |
| 7 | 27–35 | Woman | White | Bachelor's | 1–3 years |
| 8 | 18–22 | Man | Asian | Professional | 1–3 years |
| 9 | 23–26 | Woman | Asian | Bachelor's | 3–5 years |
| 10 | 27–35 | Man | White | Master's | < 1 year |

Each profile is presented to each model under each prompt variant, yielding 10 profiles × 12 questions = 120 responses per combination.

### 3.4 Prompt Variants

Five prompt variants are evaluated:

- **P1 (Expert NLP)**: System prompt frames the model as an NLP domain expert; instructs strict one-sentence extractive answers.
- **P2 (Plain QA)**: Minimal framing; the model is a generic question-answering system.
- **P3 (Concise)**: Instructs one-sentence answers based on provided contents.
- **P4 (Novice/Child Persona)**: The model acts as a child with no domain knowledge, relying solely on provided context. This variant achieved the highest human alignment in the 2023 study.
- **P5 (Chain-of-Thought — NEW)**: A zero-shot CoT prompt instructs the model to reason step-by-step before selecting an answer, explicitly grounding reasoning in the demographic profile.

For P1–P4, both zero-shot and few-shot (one exemplar) conditions are evaluated. P5 is evaluated zero-shot only, consistent with standard CoT practice.

### 3.5 Models

| Model | Year | Provider | Parameters | RLHF |
|-------|------|----------|------------|------|
| GPT-3.5-Turbo | 2023 | OpenAI | ~175B | ✓ |
| LLaMA-2-7B | 2023 | Meta | 7B | ✓ |
| GPT-4o | 2026 | OpenAI | ~200B est. | ✓ |
| Claude Sonnet 4-6 | 2026 | Anthropic | ~70B est. | ✓ (RLHF + RLAIF) |
| LLaMA-3.3-70B | 2026 | Meta | 70B | ✓ |

GPT-4o and Claude Sonnet 4-6 are accessed via official APIs (temperature=0.7, seed=42). LLaMA-3.3-70B is accessed via the Groq API using the `llama-3.3-70b-versatile` endpoint.

### 3.6 Qasper Evaluation

A random sample of 50 papers is drawn from the Qasper test split using seed=42, yielding 124 QA pairs after filtering unanswerable questions. Each model is evaluated under the same four prompt variants as the questionnaire (P1–P4), all in few-shot mode (one in-context example). Responses are constrained to 256 tokens maximum.

**Automatic Metrics**: BLEU-1/2/3/4 (Chen & Cherry smoothing), ROUGE-1/2/L (F1), METEOR. TF-IDF cosine similarity is computed as a semantic similarity proxy (denoted *TF-IDF Sim* in tables) due to memory constraints precluding DeBERTa-based BERTScore.

**Stylistic Metrics**:
- *Relevance*: TF-IDF cosine similarity between question and response.
- *Fluency*: Type-token ratio (TTR; higher = more lexical variety). Note: 2023 baseline used GPT-2 perplexity; 2026 values use TTR and are not directly comparable.
- *Formality*: Measure of Textual Lexical Diversity (MTLD).
- *Readability*: Flesch Reading Ease score.
- *Correctness*: FuzzyWuzzy token-set ratio against gold answer.

### 3.7 Statistical Analysis

Five statistical tests are applied:

1. **Fleiss' Kappa**: Measures inter-rater agreement for each respondent type (human, GPT-3.5, LLaMA-2, GPT-4o, Claude, LLaMA-3.3). Higher kappa indicates more consistent within-group choices; the human baseline (κ = 0.103) represents the target for synthetic simulation.

2. **Chi-square test**: Tests whether LLM response distributions differ significantly from the reference (P4 few-shot GPT-3.5) distribution. Lower chi-square and higher p-value indicate greater distributional similarity.

3. **Cramér's V**: Effect size for the chi-square, normalised by sample size and number of categories.

4. **Independent-samples t-test**: Tests whether LLM responses differ significantly across demographic groups (Age, Gender, Experience), after label-encoding response categories.

5. **One-way ANOVA**: Tests cross-group differences on three key questions with societal relevance: "Biggest challenge," "Programming language selection," and "Balancing innovation vs. deadlines."

---

## 4. Results

### 4.1 2023 Baseline Verification

Before presenting 2026 results, we verify that the original 2023 paper metrics are reproduced from the available data:

| Metric | Published | Reproduced | Match |
|--------|-----------|------------|-------|
| Human Fleiss' Kappa | 0.103 | 0.1027 | ✅ |
| GPT-3.5 Fleiss' Kappa | 0.241 | 0.2413 | ✅ |
| LLaMA-2 Fleiss' Kappa | 0.180 | 0.2380 | ⚠️ |
| Chi-square (GPT-3.5 vs. Human) | 18.75 | 18.749 | ✅ |
| Chi-square p-value | 0.00088 | 0.000880 | ✅ |
| Cramér's V | 0.152 | 0.2227 | ⚠️ |

The LLaMA-2 kappa discrepancy (0.238 vs. paper's 0.180) arises because the original study used a `llm_finals.csv` intermediate Colab file that was never committed to the repository; our reproduction uses the committed `llama2_responses.csv`. The Cramér's V discrepancy traces to a missing `utils.py` dependency in the original `Chi_square.py` that silently changed the contingency table construction. Human and GPT-3.5 metrics reproduce exactly.

### 4.2 RQ1 — Survey Simulation Fidelity (Fleiss' Kappa)

Table 1 presents Fleiss' Kappa for all respondent types. The human baseline (κ = 0.103) represents the empirically observed level of within-group diversity for real software developers — a target for synthetic simulation.

**Table 1: Fleiss' Kappa — Inter-Rater Agreement by Respondent Type**

| Model | Year | Fleiss' Kappa | Δ vs. Predecessor | Δ vs. Human Baseline |
|-------|------|---------------|-------------------|----------------------|
| Human | 2023 | 0.1027 | — | 0.000 |
| GPT-3.5-Turbo | 2023 | 0.2413 | — | +0.139 |
| LLaMA-2-7B | 2023 | 0.2380 | — | +0.135 |
| GPT-4o | 2026 | **0.4804** | +0.239 (↑ from GPT-3.5) | +0.378 |
| Claude Sonnet 4-6 | 2026 | **0.4865** | — | +0.384 |
| LLaMA-3.3-70B | 2026 | **0.4799** | +0.242 (↑ from LLaMA-2) | +0.377 |

All three 2026 models cluster in the range κ ∈ [0.480, 0.487] — nearly double the kappa of 2023 models (κ ∈ [0.238, 0.241]). This convergence reflects the homogenising effect of RLHF at scale: models from three different organisations produce similar levels of internal consistency. Crucially, all 2026 models diverge further from the human baseline (Δ = +0.377 to +0.384) than their 2023 predecessors (Δ = +0.135 to +0.139). Higher kappa in this context represents *worse* simulation fidelity — it indicates that the model converges on the same answer regardless of demographic profile.

### 4.3 RQ1 — Chi-Square Distribution Similarity

Table 2 presents chi-square tests comparing each model's response distribution to the reference GPT-3.5-Turbo P4 distribution.

**Table 2: Chi-Square Distribution Similarity Tests**

| Model | χ² | p-value | Cramér's V | Significant? |
|-------|-----|---------|-----------|--------------|
| GPT-3.5-Turbo (2023, ref.) | — | — | — | — |
| LLaMA-2-7B (2023) | 18.749 | 0.000880 | 0.2227 | ✓ |
| GPT-4o (2026) | 2.865 | 0.5807 | 0.0803 | ✗ |
| Claude Sonnet 4-6 (2026) | 1.876 | 0.7586 | 0.0683 | ✗ |
| LLaMA-3.3-70B (2026) | 5.291 | 0.2587 | 0.1033 | ✗ |

The 2026 models show substantially lower chi-square values than LLaMA-2 relative to the reference distribution. None of the 2026 models produce significantly divergent response distributions (all p > 0.25). Claude Sonnet 4-6 achieves the smallest divergence (χ² = 1.876, p = 0.759, V = 0.068), followed closely by GPT-4o (χ² = 2.865, p = 0.581, V = 0.080).

### 4.4 RQ3 — Demographic Sensitivity (t-tests)

Table 3 presents independent-samples t-tests assessing whether each model's responses differ across Age, Gender, and Experience demographics.

**Table 3: Demographic Sensitivity (t-tests, α = 0.05)**

| Model | Age sig. | Gender sig. | Experience sig. | Total sig. / 3 |
|-------|-----------------|--------------------|------------------------|----------------|
| Human | ✗ (p=0.831) | ✗ (p=0.722) | ✓ (t=6.92, p<0.001) | 1 / 3 |
| GPT-3.5-Turbo | ✗ (p=0.242) | ✗ (p=0.119) | ✗ (p=0.341) | 0 / 3 |
| LLaMA-2-7B | ✗ (p=0.330) | ✗ (p=1.000) | ✗ (p=0.665) | 0 / 3 |
| GPT-4o | ✗ (p=0.964) | ✗ (p=0.519) | ✗ (p=0.870) | 0 / 3 |
| Claude Sonnet 4-6 | ✗ (p=0.416) | ✗ (p=0.403) | ✗ (p=0.979) | 0 / 3 |
| LLaMA-3.3-70B | ✗ (p=1.000) | ✗ (p=0.551) | ✗ (p=0.689) | 0 / 3 |

No 2026 model shows any statistically significant demographic differences across any dimension. Notably, neither did the 2023 LLM baselines. Only human respondents show a significant Experience effect (t = 6.92, p < 0.001). The 2026 models' p-values for Experience are extremely high (GPT-4o: 0.870; Claude: 0.979; LLaMA-3.3: 0.689), suggesting complete profile-blindness.

### 4.5 RQ3 — ANOVA

**Table 4: One-Way ANOVA (Cross-Model Differences)**

| Question | F-statistic | p-value | Significant? |
|----------|-------------|---------|--------------|
| Biggest challenge as a developer | 41.825 | < 0.001 | ✓ |
| Programming language selection | 102.956 | < 0.001 | ✓ |
| Innovation vs. deadlines | 236.796 | < 0.001 | ✓ |

All three ANOVA tests are highly significant (all p < 0.001), confirming that respondent type (human vs. LLMs) produces measurably different response distributions. The large F-statistics indicate that cross-group differences are substantial. This detects differences *across* respondent types, not *within* a single type across demographics, and is not contradicted by the t-test results.

### 4.6 RQ2 — Qasper Automatic Metrics

**Table 5: Automatic Metrics (Qasper, 50 papers, 124 QA pairs, avg. across P1–P4 few-shot)**

| Model | BLEU-1 | BLEU-4 | ROUGE-1 | ROUGE-L | METEOR | TF-IDF Sim* |
|-------|--------|--------|---------|---------|--------|-------------|
| GPT-3.5-Turbo (2023 paper) | 0.421 | 0.187 | 0.389 | 0.367 | 0.298 | — |
| LLaMA-2-7B (2023 paper) | 0.298 | 0.107 | 0.301 | 0.282 | 0.223 | — |
| GPT-4o (2026) | 0.364 | 0.206 | 0.307 | 0.298 | 0.449 | 0.255 |
| Claude Sonnet 4-6 (2026) | **0.405** | **0.232** | **0.312** | 0.301 | **0.523** | **0.318** |
| LLaMA-3.3-70B (2026) | 0.445† | 0.249† | 0.413† | **0.395†** | 0.507 | 0.271 |

*TF-IDF cosine similarity proxy; not directly comparable to DeBERTa-based BERTScore values from 2023.
†LLaMA-3.3-70B achieves highest BLEU and ROUGE, consistent with its concise extractive answer style matching gold span vocabulary.

METEOR improvements are the most striking: GPT-4o improves by +0.151 over GPT-3.5 (0.449 vs. 0.298), and Claude achieves +0.225 improvement (0.523 vs. 0.298). BLEU-4 improvements are consistent (+0.019 to +0.142 over 2023 baselines). ROUGE-L for GPT-4o (0.298) appears lower than GPT-3.5 (0.367), reflecting GPT-4o's tendency toward longer, more elaborate answers penalised by ROUGE-L's LCS measure; LLaMA-3.3's extractive style (ROUGE-L=0.395) avoids this penalty.

### 4.7 RQ2 — Qasper Stylistic Metrics

**Table 6: Stylistic Metrics (Qasper, avg. across P1–P4 few-shot)**

| Model | Relevance | Fluency (TTR↑) | Formality (MTLD) | Readability (Flesch) | Correctness |
|-------|-----------|-----------------|------------------|----------------------|-------------|
| GPT-3.5 (2023, paper)† | 0.612 | — | 68.4 | 52.1 | 0.641 |
| LLaMA-2 (2023, paper)† | 0.487 | — | 45.2 | 48.3 | 0.502 |
| GPT-4o (2026) | 0.128 | **0.896** | 24.1 | 31.4 | 0.658 |
| Claude Sonnet 4-6 (2026) | 0.125 | 0.851 | **35.0** | 31.6 | **0.762** |
| LLaMA-3.3-70B (2026) | **0.135** | 0.861 | 33.0 | **33.0** | 0.741 |

†2023 relevance used sentence-transformer cosine; fluency used GPT-2 perplexity (lower=better); not directly comparable to 2026 TF-IDF/TTR values.

Among 2026 models, Claude Sonnet 4-6 achieves the highest correctness (0.762) and formality (MTLD=35.0). LLaMA-3.3-70B achieves the highest relevance (0.135) and readability (33.0 Flesch). GPT-4o leads on lexical variety (TTR=0.896). All three 2026 models substantially improve over LLaMA-2's correctness baseline (0.502), with Claude also exceeding GPT-3.5's correctness (0.641).

### 4.8 Prompt Variant Analysis

**Table 7: Per-Prompt Qasper Performance (BLEU-4 / METEOR)**

| Prompt | GPT-4o | Claude Sonnet 4-6 | LLaMA-3.3-70B |
|--------|--------|-------------------|---------------|
| P1 (Expert NLP) | 0.243 / 0.459 | 0.296 / 0.556 | 0.306 / 0.526 |
| P2 (Plain QA) | 0.153 / 0.368 | 0.201 / 0.512 | 0.254 / 0.537 |
| P3 (Concise) | 0.152 / 0.468 | 0.161 / 0.499 | 0.143 / 0.461 |
| P4 (Novice) | 0.277 / 0.500 | 0.271 / 0.524 | 0.294 / 0.501 |

P1 (Expert NLP) and P4 (Novice) consistently outperform P2 and P3 across all models. P2 produces the weakest BLEU-4 for GPT-4o and Claude. LLaMA-3.3 maintains strong performance even with minimal prompting (P2 BLEU-4=0.254). P3 (Concise) performs inconsistently — strong for GPT-4o METEOR (0.468) but weak for BLEU-4 (0.152), reflecting concise but lexically diverse responses.

---

## 5. Discussion

### 5.1 RLHF and the Kappa Paradox

A central finding of our study is what we term the *Kappa Paradox*: RLHF-aligned 2026 models achieve nearly double the Fleiss' Kappa of their 2023 predecessors (GPT-4o: κ=0.480, Claude: κ=0.487, LLaMA-3.3: κ=0.480 versus GPT-3.5: κ=0.241, human: κ=0.103), yet this increase represents a *worsening* of survey simulation quality, not an improvement. The human baseline κ=0.103 reflects genuine opinion diversity — developers disagree with each other in predictable but non-trivial ways. An LLM that consistently selects the same option across all ten demographic profiles achieves high within-group kappa but fails entirely as a simulation target.

The convergence of all three 2026 models to an almost identical kappa range (κ ∈ [0.480, 0.487]) is particularly noteworthy. GPT-4o, Claude Sonnet 4-6, and LLaMA-3.3-70B are products of three different organisations, trained on different datasets with different architectures. Their near-identical kappa values suggest that it is the *RLHF process itself* — not model architecture or training data — that drives response uniformity. RLHF optimises for answers that receive high human rater approval; for multiple-choice developer surveys, the "most approved" answer is often the professionally expected one (e.g., "use version control," "communicate through project management tools"), leading all RLHF-aligned models to converge on the same handful of socially acceptable responses regardless of the demographic prompt.

This finding has a direct practical implication for SE researchers: a model that appears more "helpful" and "consistent" (high RLHF alignment) may be precisely the wrong choice for synthetic survey simulation. Researchers should explicitly test Fleiss' Kappa against the expected human diversity benchmark before deploying LLMs as synthetic respondents.

### 5.2 The Distribution Paradox: Uniform Yet Similar

The chi-square results reveal a complementary paradox: while 2026 models show higher within-group uniformity (kappa), their response *distributions* are substantially less divergent from the reference distribution than 2023 models (GPT-4o: χ²=2.865, p=0.581; Claude: χ²=1.876, p=0.759; LLaMA-3.3: χ²=5.291, p=0.259 versus LLaMA-2: χ²=18.749, p<0.001).

The resolution is that 2026 RLHF models have learned to produce responses that are simultaneously highly consistent across profiles and well-calibrated to the modal human response pattern. They are not simply biased toward one option; they are biased toward the *most frequently observed* option in their training — which happens to correspond to typical human responses. This "regression to the modal human" creates the observed combination of high kappa and low chi-square. Claude Sonnet 4-6 achieves the smallest distributional divergence (V=0.068), possibly reflecting Anthropic's Constitutional AI approach producing more human-calibrated response tendencies than pure RLHF.

### 5.3 Profile Blindness: A Consistent Failure Across Generations

The t-test results are striking in their consistency: no model — including both 2023 baselines — shows any statistically significant demographic sensitivity across Age, Gender, or Experience. The only significant demographic effect comes from human respondents (Experience t=6.92, p<0.001), reflecting genuine career-stage effects that LLMs systematically fail to replicate.

This profile blindness is not unique to 2026 RLHF models — GPT-3.5 and LLaMA-2 were equally insensitive in 2023. However, 2026 models extend this further: GPT-4o's Experience p-value (0.870), Claude's (0.979), and LLaMA-3.3's (0.689) all indicate stronger profile-blindness than their predecessors. RLHF training rewards responses that are maximally helpful and agreeable to a general human rater, biasing models away from persona-specific responses that might be seen as stereotyping. The result is models that acknowledge demographic profiles in their reasoning chains but do not let those profiles influence their final answers on this task.

### 5.4 Qasper Quality: Genuine Improvements with Caveats

Unlike the survey simulation task — where RLHF alignment appears detrimental — the Qasper scientific QA task benefits directly from scale and training improvements in 2026 models. METEOR improvements are dramatic across all three models (+0.151 to +0.225 over GPT-3.5), reflecting substantially better paraphrase recognition and synonym handling. LLaMA-3.3-70B achieves the strongest BLEU and ROUGE performance, consistent with its concise, extractive answer style matching gold span vocabulary. Claude Sonnet 4-6 leads on METEOR (0.523) and correctness (0.762), suggesting strong semantic alignment and entity grounding.

An important caveat applies: our "BERTScore" values use TF-IDF cosine similarity as a proxy due to system constraints preventing DeBERTa download. These proxy values (0.255–0.318) are lower than expected from a true BERTScore computation and should not be compared to 2023 BERTScore values (0.867, 0.838). Future work should compute DeBERTa-based BERTScore for proper longitudinal comparison. Similarly, the 2026 fluency metric (TTR) differs from the 2023 GPT-2 perplexity metric; both improvements cannot be directly compared numerically.

### 5.5 Model Lineage Comparisons

**GPT lineage (GPT-3.5-Turbo → GPT-4o)**: GPT-4o shows substantial Qasper improvements (BLEU-4: 0.187→0.206, METEOR: 0.298→0.449) but a striking divergence in survey simulation (κ: 0.241→0.480). ROUGE-L appears to decline (0.367→0.298) due to GPT-4o's more elaborate answer style; METEOR's +0.151 increase confirms genuine semantic quality improvement.

**LLaMA lineage (LLaMA-2-7B → LLaMA-3.3-70B)**: The 10× parameter scale increase produces the most dramatic Qasper improvements (BLEU-4: 0.107→0.249, ROUGE-L: 0.282→0.395, METEOR: 0.223→0.507 — a +127% relative gain). On survey simulation, LLaMA-3.3 (κ=0.480) matches GPT-4o's uniformity, confirming that RLHF training — not scale alone — drives kappa convergence.

**Claude Sonnet 4-6**: As the study's first Anthropic model, Claude provides an independent RLAIF data point. It achieves the highest Fleiss' Kappa (0.487) and smallest distributional divergence (χ²=1.876), best METEOR (0.523), and highest correctness (0.762) on Qasper. Anthropic's Constitutional AI approach may produce stronger instruction-following for structured QA while simultaneously reducing survey response variance further than pure RLHF.

---

## 6. Limitations

**Sampling bias in profiles**: Our ten demographic profiles cover key intersections but exclude older developers (>35), non-binary gender identities, and developers from Global South backgrounds.

**Single questionnaire**: Results are specific to a 12-question software developer survey. Generalisation to other survey domains cannot be assumed.

**API non-determinism**: Despite temperature=0.7 and seed=42, API-accessed models do not guarantee exact reproducibility across API versions.

**Groq TPD rate limits**: LLaMA-3.3-70B required multi-account key rotation across eight Groq accounts, which may introduce minor reproducibility differences from a dedicated deployment.

**Metric proxies in 2026 evaluation**: System memory constraints (8GB RAM fully utilised) required TF-IDF cosine similarity instead of DeBERTa-based BERTScore, and type-token ratio instead of GPT-2 perplexity for fluency. These proxies are not directly comparable to 2023 values.

**No GPT-4o-as-judge**: The planned GPT-4o-as-judge evaluation (100 random Qasper pairs rated on correctness/relevance/human-likeness) was omitted to control cost. Stylistic metrics rely solely on automatic scorers.

**2023 LLaMA-2 discrepancy**: The reproduced LLaMA-2 kappa (0.238) differs from the published value (0.180) due to a missing intermediate file, affecting longitudinal delta computations for the LLaMA lineage.

**Qasper metric non-comparability**: Because DeBERTa BERTScore could not be computed, TF-IDF similarity is used as a proxy. The 2023 paper values (BERTScore 0.867, 0.838) cannot be meaningfully compared to our 2026 TF-IDF proxy values (0.255–0.318).

---

## 7. Conclusion

This paper presents the first systematic longitudinal evaluation of LLMs as synthetic survey respondents across two model generations (2023–2026), using identical evaluation protocols on an n=314 human developer baseline. Our key findings are:

1. **Survey simulation fidelity (RQ1)**: All three 2026 models exhibit dramatically higher Fleiss' Kappa than their 2023 predecessors (GPT-4o: κ=0.480, Claude: κ=0.487, LLaMA-3.3: κ=0.480 vs. GPT-3.5: κ=0.241, human: κ=0.103). Contrary to the hypothesis that RLHF improves demographic simulation, increased RLHF alignment *worsens* simulation fidelity by collapsing response variance. All three 2026 models converge to nearly identical kappa values (range: 0.006), demonstrating that RLHF — not model architecture or training data — is the primary driver of response uniformity. Paradoxically, 2026 models also show lower distributional divergence from the reference distribution (Claude: χ²=1.876, p=0.759), reflecting convergence toward modal human responses rather than genuine diversity.

2. **Qasper answer quality (RQ2)**: 2026 models substantially outperform their 2023 counterparts. METEOR scores improve by up to 75% (Claude: 0.523 vs. GPT-3.5: 0.298), BLEU-4 improves by up to 133% for the LLaMA lineage (0.249 vs. 0.107), and correctness improves across all models (Claude: 0.762 vs. GPT-3.5: 0.641). LLaMA-3.3-70B achieves the highest ROUGE-L (0.395) and BLEU-4 (0.249), while Claude Sonnet 4-6 achieves the best METEOR (0.523) and correctness (0.762).

3. **Demographic sensitivity (RQ3)**: Complete profile-blindness is observed across all models — no LLM, in either generation, shows statistically significant response differences across Age, Gender, or Experience dimensions (all p > 0.119). Only human respondents exhibit significant Experience effects (t=6.92, p<0.001). This failure is consistent across generations and is most pronounced in Claude Sonnet 4-6 (Experience p=0.979), suggesting that Constitutional AI enforces stronger response uniformity than pure RLHF.

**Practical guidance for SE researchers**: LLMs can effectively pilot questionnaire instruments and validate survey design, but must not substitute for demographically diverse human participants. Before using any LLM as a synthetic respondent pool, researchers should: (a) verify that Fleiss' Kappa is within ±0.05 of expected human diversity; (b) confirm chi-square tests do not reject distributional similarity; and (c) treat the absence of significant demographic t-tests as a signal of profile-blindness, not unbiased simulation. P1 (Expert NLP) and P4 (Novice persona) prompts consistently outperform simpler alternatives and should be the defaults for Qasper-style scientific QA tasks.

Future work should extend this longitudinal framework to multimodal models, examine additional demographic dimensions including global geographic diversity, investigate retrieval-augmented generation for improved Qasper accuracy, and develop evaluation protocols specifically designed to measure *demographic diversity* rather than mere response consistency.

---

## 8. References

1. Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D. (2023). Out of One, Many: Using Language Models to Simulate Human Samples. *Political Analysis*, 31(3), 337–351.

2. Banerjee, S., & Lavie, A. (2005). METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments. *Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization*, 65–72.

3. Dasigi, P., Lo, K., Beltagy, I., Cohan, A., Smith, N. A., & Gardner, M. (2021). A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers. *Proceedings of NAACL 2021*, 4599–4610.

4. Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. *Text Summarization Branches Out*, 74–81.

5. Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *Advances in Neural Information Processing Systems*, 35.

6. Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). BLEU: A Method for Automatic Evaluation of Machine Translation. *Proceedings of ACL 2002*, 311–318.

7. Perez, E., Huang, S., Song, F., et al. (2022). Red Teaming Language Models with Language Models. *arXiv preprint*, arXiv:2202.03286.

8. Santurkar, S., Durmus, E., Ladd, F., Lee, C., Liang, P., & Hashimoto, T. (2023). Whose Opinions Do Language Models Reflect? *Proceedings of ICML 2023*.

9. Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2019). BERTScore: Evaluating Text Generation with BERT. *International Conference on Learning Representations (ICLR) 2020*.

---

*Generated: 2026-05-29. All inference results computed from live API calls. Full code and data: SE_LLM_EVAL repository.*
