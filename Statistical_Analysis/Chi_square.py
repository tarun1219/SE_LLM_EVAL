import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'Datasets')

gpt35_data  = pd.read_csv(os.path.join(DATA_DIR, 'gpt3.5_responses.csv'))
llama2_data = pd.read_csv(os.path.join(DATA_DIR, 'llama2_responses.csv'))
human_data  = pd.read_csv(os.path.join(DATA_DIR, 'survey_responses.csv'))

# 2026 model responses — loaded if CSVs exist
NEW_MODELS = {}
for model_file, label in [
    ('gpt4o_responses.csv', 'gpt4o'),
    ('claude_responses.csv', 'claude'),
    ('llama31_responses.csv', 'llama31'),
]:
    path = os.path.join(DATA_DIR, model_file)
    if os.path.exists(path):
        NEW_MODELS[label] = pd.read_csv(path)

# Human responses use 'Option' column; LLM responses use 'Answer' / 'Option IDs'
human_data = human_data.rename(columns={'Option': 'Answer'})

# Build per-model option counts (handles comma-separated multi-select)
gpt35_option_counts  = gpt35_data['Option IDs'].str.get_dummies(sep=', ').sum()
llama2_option_counts = llama2_data['Answer'].str.get_dummies(sep=', ').sum()
human_option_counts  = human_data['Answer'].value_counts()

option_counts_comparison = pd.DataFrame({
    'GPT-3.5': gpt35_option_counts,
    'LLaMA-2': llama2_option_counts,
    'Human':   human_option_counts,
}).fillna(0)

# Add 2026 models if available
for label, df in NEW_MODELS.items():
    col = 'Option IDs' if 'Option IDs' in df.columns else 'Answer'
    option_counts_comparison[label] = df[col].str.get_dummies(sep=', ').sum()
option_counts_comparison = option_counts_comparison.fillna(0)


def run_chi_square_cramers(contingency_table: pd.DataFrame, label: str = "") -> None:
    chi2, p, dof, expected = chi2_contingency(contingency_table.T)
    n = contingency_table.values.sum()
    k = min(contingency_table.shape)
    cramers_v = np.sqrt(chi2 / (n * (k - 1))) if k > 1 else 0.0

    print(f"[{label}]")
    print(f"  Chi-Square Statistic : {chi2:.4f}")
    print(f"  p-value              : {p:.4f}")
    print(f"  Degrees of Freedom   : {dof}")
    print(f"  Cramér's V           : {cramers_v:.4f}\n")


# 2023 baseline (3 groups)
baseline_cols = ['GPT-3.5', 'LLaMA-2', 'Human']
run_chi_square_cramers(option_counts_comparison[baseline_cols], label='2023_baseline')

# Longitudinal (all groups, only if new model CSVs present)
if NEW_MODELS:
    run_chi_square_cramers(option_counts_comparison, label='longitudinal')
