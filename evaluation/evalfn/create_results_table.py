"""
Create CiK-style Results Table for Glucose Forecasting
========================================================

Generates a comprehensive results table similar to CiK benchmark Table 1,
showing model performance across multiple metrics and stratified by context types.

Table structure:
- Rows: Models grouped by type (Foundation Models, Statistical, etc.)
- Columns: Overall metrics + stratified metrics by context type
- Values: mean ± std with bold for best performers

Usage:
    python evaluation/eval_fn/create_results_table.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    """Load clinical metrics CSV and prepare for analysis."""
    df = pd.read_csv(csv_path)

    # Filter valid rows
    valid = df[df['glucose_rcrps'].notna()].copy()

    return valid


def extract_context_type(task_name: str) -> str:
    """
    Extract context type from task name.

    Examples:
        EventCGMTask_Base → No Context
        EventCGMTask_D1_Age40_Diet_Ontime_DetailedEvent → Diet (Detailed)
        EventCGMTask_D2_Age65_Exercise_Ontime_Profile → Exercise (Profile)
    """
    if 'Base' in task_name:
        return 'No Context'

    # Extract event type
    if 'Diet' in task_name:
        event_type = 'Diet'
    elif 'Exercise' in task_name:
        event_type = 'Exercise'
    elif 'Med' in task_name:
        event_type = 'Medication'
    else:
        event_type = 'Unknown'

    # Extract context level
    if 'NoCtx' in task_name:
        context_level = 'No Context'
    elif 'DetailedEvent' in task_name:
        context_level = 'Detailed Event'
    elif 'MediumEvent' in task_name:
        context_level = 'Medium Event'
    elif 'Profile' in task_name:
        context_level = 'Profile Only'
    else:
        context_level = 'Unknown'

    if context_level == 'No Context':
        return 'No Context'
    else:
        return f"{event_type} ({context_level})"


def extract_patient_demographic(task_name: str) -> str:
    """
    Extract patient demographic from task name.

    Examples:
        EventCGMTask_D1_Age40_Diet → D1, Age 40
        EventCGMTask_D2_Age65_Exercise → D2, Age 65
    """
    if 'Base' in task_name:
        return 'Mixed'

    # Extract diabetes type
    if 'D1' in task_name:
        diabetes_type = 'Type 1'
    elif 'D2' in task_name:
        diabetes_type = 'Type 2'
    else:
        diabetes_type = 'Mixed'

    # Extract age group
    if 'Age18' in task_name:
        age_group = 'Age 18-30'
    elif 'Age40' in task_name:
        age_group = 'Age 40-50'
    elif 'Age65' in task_name:
        age_group = 'Age 65+'
    else:
        age_group = 'Mixed'

    if diabetes_type == 'Mixed':
        return 'Mixed'
    else:
        return f"{diabetes_type}, {age_group}"


def compute_stratified_metrics(
    df: pd.DataFrame,
    stratify_by: str = 'context'
) -> pd.DataFrame:
    """
    Compute metrics stratified by context type or patient demographics.

    Args:
        df: DataFrame with model results
        stratify_by: 'context', 'demographic', or 'event_type'

    Returns:
        DataFrame with columns: model, metric_mean, metric_std, metric_rank for each stratum
    """
    # Add stratification column
    if stratify_by == 'context':
        df['stratum'] = df['task'].apply(extract_context_type)
    elif stratify_by == 'demographic':
        df['stratum'] = df['task'].apply(extract_patient_demographic)
    else:
        raise ValueError(f"Unknown stratification: {stratify_by}")

    results = []

    for model in df['model'].unique():
        model_data = df[df['model'] == model]

        # Overall metrics
        overall_metrics = {
            'model': model,
            'n_instances': len(model_data),
        }

        for metric in ['crps', 'glucose_rcrps', 'weighted_term', 'clarke_ab', 'mae']:
            if metric in model_data.columns:
                values = model_data[metric].dropna()
                overall_metrics[f'{metric}_mean'] = values.mean()
                overall_metrics[f'{metric}_std'] = values.std() / np.sqrt(len(values))  # SEM

        # Stratified metrics
        for stratum in df['stratum'].unique():
            stratum_data = model_data[model_data['stratum'] == stratum]

            if len(stratum_data) == 0:
                continue

            for metric in ['crps', 'glucose_rcrps', 'weighted_term', 'clarke_ab']:
                if metric in stratum_data.columns:
                    values = stratum_data[metric].dropna()
                    if len(values) > 0:
                        overall_metrics[f'{metric}_{stratum}_mean'] = values.mean()
                        overall_metrics[f'{metric}_{stratum}_std'] = values.std() / np.sqrt(len(values))

        results.append(overall_metrics)

    results_df = pd.DataFrame(results)

    # Compute ranks (lower is better for CRPS, MAE; higher is better for Clarke A+B)
    for metric in ['crps', 'glucose_rcrps', 'weighted_term', 'mae']:
        if f'{metric}_mean' in results_df.columns:
            results_df[f'{metric}_rank'] = results_df[f'{metric}_mean'].rank(method='average')

    if 'clarke_ab_mean' in results_df.columns:
        results_df['clarke_ab_rank'] = results_df['clarke_ab_mean'].rank(method='average', ascending=False)

    return results_df


def categorize_model(model_name: str) -> str:
    """Categorize model into groups for table organization."""
    if 'chronos' in model_name.lower():
        return 'TS Foundation Models'
    elif 'lag-llama' in model_name.lower() or 'lagllama' in model_name.lower():
        return 'TS Foundation Models'
    elif 'unitime' in model_name.lower():
        return 'TS Foundation Models'
    elif 'moirai' in model_name.lower():
        return 'TS Foundation Models'
    elif 'timellm' in model_name.lower():
        return 'Multimodal Models'
    elif 'gpt' in model_name.lower() or 'claude' in model_name.lower() or 'llama' in model_name.lower():
        return 'LLM-based Models'
    elif 'arima' in model_name.lower() or 'ets' in model_name.lower():
        return 'Statistical Models'
    else:
        return 'Other Models'


def format_value_with_error(mean: float, std: float, is_best: bool = False) -> str:
    """Format value as 'mean ± std' with optional bold for best."""
    formatted = f"{mean:.3f} ± {std:.3f}"
    if is_best:
        formatted = f"\\textbf{{{formatted}}}"
    return formatted


def create_latex_table(
    results_df: pd.DataFrame,
    metrics: List[str],
    stratifications: List[str],
    output_path: Path
):
    """
    Create LaTeX table similar to CiK benchmark Table 1.

    Args:
        results_df: DataFrame with computed metrics
        metrics: List of metrics to include (e.g., ['crps', 'glucose_rcrps', 'weighted_term'])
        stratifications: List of stratification categories to show
        output_path: Path to save LaTeX file
    """
    # Group models by category
    results_df['category'] = results_df['model'].apply(categorize_model)

    # Sort by category and then by primary metric
    primary_metric = metrics[0]
    results_df = results_df.sort_values(['category', f'{primary_metric}_mean'])

    # Start LaTeX table
    latex = []
    latex.append("\\begin{table*}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Results of glucose forecasting models. Starting from the left, the first column shows the average Glucose-RCRPS across all tasks. The second column shows the rank of each method w.r.t. other models, averaged over all tasks. The remaining columns show the average Glucose-RCRPS stratified by context types. All averages are weighted and accompanied by standard errors. Lower is better and the best averages are in bold.}")
    latex.append("\\label{tab:glucose_results}")

    # Table header
    n_cols = 2 + len(stratifications)  # model, avg metric, rank, stratifications
    latex.append(f"\\begin{{tabular}}{{l{'c' * (n_cols - 1)}}}")
    latex.append("\\toprule")

    # Column headers
    header = "\\textsc{Model} & \\textsc{Average} & \\textsc{Average} & " + \
             " & ".join([f"\\textsc{{{s}}}" for s in stratifications]) + " \\\\"
    latex.append(header)

    subheader = " & \\textsc{Glucose-RCRPS} & \\textsc{Rank} & " + \
                " & ".join(["" for _ in stratifications]) + " \\\\"
    latex.append(subheader)
    latex.append("\\midrule")

    # Find best values for each column
    best_values = {}
    best_values['glucose_rcrps_mean'] = results_df['glucose_rcrps_mean'].min()
    for strat in stratifications:
        col = f'glucose_rcrps_{strat}_mean'
        if col in results_df.columns:
            best_values[col] = results_df[col].min()

    # Add rows grouped by category
    current_category = None
    for _, row in results_df.iterrows():
        # Add category header
        if row['category'] != current_category:
            if current_category is not None:
                latex.append("\\midrule")
            latex.append(f"\\textsc{{{row['category']}}}")
            current_category = row['category']

        # Model name
        model_name = row['model'].replace('_', '\\_')

        # Average Glucose-RCRPS
        is_best = abs(row['glucose_rcrps_mean'] - best_values['glucose_rcrps_mean']) < 1e-6
        avg_grcrps = format_value_with_error(
            row['glucose_rcrps_mean'],
            row['glucose_rcrps_std'],
            is_best
        )

        # Average rank
        avg_rank = f"{row['glucose_rcrps_rank']:.2f}"
        if is_best:
            avg_rank = f"\\textbf{{{avg_rank}}}"

        # Stratified values
        strat_values = []
        for strat in stratifications:
            col_mean = f'glucose_rcrps_{strat}_mean'
            col_std = f'glucose_rcrps_{strat}_std'

            if col_mean in row.index and not pd.isna(row[col_mean]):
                is_best_strat = abs(row[col_mean] - best_values[col_mean]) < 1e-6
                val = format_value_with_error(
                    row[col_mean],
                    row[col_std],
                    is_best_strat
                )
                strat_values.append(val)
            else:
                strat_values.append("---")

        # Combine into row
        row_str = f"  {model_name} & {avg_grcrps} & {avg_rank} & " + \
                  " & ".join(strat_values) + " \\\\"
        latex.append(row_str)

    # Close table
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table*}")

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))

    print(f"LaTeX table saved to: {output_path}")
    return '\n'.join(latex)


def create_comparison_table(
    results_df: pd.DataFrame,
    output_path: Path
):
    """
    Create comparison table showing CRPS vs Glucose-RCRPS vs Weighted-term.

    Similar to showing how different metrics rank models differently.
    """
    # Select top 10 models by Glucose-RCRPS
    top_models = results_df.nsmallest(10, 'glucose_rcrps_mean')

    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Comparison of metrics: Standard CRPS, Glucose-RCRPS, and Clarke-weighted CRPS. Lower is better for all metrics. Bold indicates best value per column.}")
    latex.append("\\label{tab:metric_comparison}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\toprule")
    latex.append("\\textsc{Model} & \\textsc{CRPS} & \\textsc{Glucose-RCRPS} & \\textsc{Weighted CRPS} \\\\")
    latex.append("\\midrule")

    # Find best values
    best_crps = top_models['crps_mean'].min()
    best_grcrps = top_models['glucose_rcrps_mean'].min()
    best_weighted = top_models['weighted_term_mean'].min()

    for _, row in top_models.iterrows():
        model = row['model'].replace('_', '\\_')

        # CRPS
        is_best = abs(row['crps_mean'] - best_crps) < 1e-6
        crps_val = format_value_with_error(row['crps_mean'], row['crps_std'], is_best)

        # Glucose-RCRPS
        is_best = abs(row['glucose_rcrps_mean'] - best_grcrps) < 1e-6
        grcrps_val = format_value_with_error(row['glucose_rcrps_mean'], row['glucose_rcrps_std'], is_best)

        # Weighted term
        is_best = abs(row['weighted_term_mean'] - best_weighted) < 1e-6
        weighted_val = format_value_with_error(row['weighted_term_mean'], row['weighted_term_std'], is_best)

        latex.append(f"  {model} & {crps_val} & {grcrps_val} & {weighted_val} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))

    print(f"Comparison table saved to: {output_path}")
    return '\n'.join(latex)


def main():
    """Generate result tables."""
    print("=" * 80)
    print("CREATING CIK-STYLE RESULTS TABLES")
    print("=" * 80)

    # Setup paths
    repo_root = Path(__file__).resolve().parent.parent.parent
    csv_path = repo_root / 'evaluation' / 'results' / 'clinical_metrics_detailed.csv'
    output_dir = repo_root / 'evaluation' / 'results'

    # Load data
    print(f"\n1. Loading data from: {csv_path}")
    df = load_and_prepare_data(csv_path)
    print(f"   Loaded {len(df)} valid rows")

    # Compute stratified metrics
    print(f"\n2. Computing stratified metrics...")
    results_df = compute_stratified_metrics(df, stratify_by='context')

    # Define stratifications for table (similar to CiK's context types)
    stratifications = [
        'No Context',
        'Diet (Detailed Event)',
        'Diet (Profile Only)',
        'Exercise (Detailed Event)',
        'Exercise (Profile Only)'
    ]

    # Create main results table
    print(f"\n3. Creating main results table...")
    latex_path = output_dir / 'glucose_results_table.tex'
    create_latex_table(
        results_df,
        metrics=['glucose_rcrps', 'weighted_term', 'crps'],
        stratifications=stratifications,
        output_path=latex_path
    )

    # Create comparison table
    print(f"\n4. Creating metric comparison table...")
    comparison_path = output_dir / 'metric_comparison_table.tex'
    create_comparison_table(results_df, comparison_path)

    print("\n" + "=" * 80)
    print("TABLE GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - {latex_path}")
    print(f"  - {comparison_path}")


if __name__ == '__main__':
    main()
