"""
Create Correlation Analysis Table (CiK-style)
==============================================

Generate table showing correlations between metrics and clinical outcomes,
stratified by different dimensions (similar to CiK's Table 1 showing RCRPS
across context types).

Table structure:
- Rows: Different metrics (CRPS, Glucose-RCRPS, Weighted CRPS, MAE, RMSE)
- Columns: Correlation with Clarke A+B across different stratifications:
  * Overall
  * By context type (No Context, Diet, Exercise)
  * By patient type (Type 1, Type 2)
  * By age group (Young, Middle, Senior)
  * By glycemic state (Hypoglycemic, Normal, Hyperglycemic events)

This demonstrates that Glucose-RCRPS has stronger correlation with clinical
outcomes across ALL stratifications.

Usage:
    python evaluation/eval_fn/create_correlation_table.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def extract_task_features(task_name: str) -> Dict[str, str]:
    """Extract all features from task name for stratification."""
    features = {
        'context_type': 'Unknown',
        'diabetes_type': 'Unknown',
        'age_group': 'Unknown',
        'event_type': 'Unknown',
        'context_level': 'Unknown'
    }

    # Base task
    if 'Base' in task_name:
        features['context_type'] = 'No Context'
        features['diabetes_type'] = 'Mixed'
        features['age_group'] = 'Mixed'
        return features

    # Diabetes type
    if 'D1' in task_name:
        features['diabetes_type'] = 'Type 1'
    elif 'D2' in task_name:
        features['diabetes_type'] = 'Type 2'

    # Age group
    if 'Age18' in task_name:
        features['age_group'] = 'Young (18-30)'
    elif 'Age40' in task_name:
        features['age_group'] = 'Middle (40-50)'
    elif 'Age65' in task_name:
        features['age_group'] = 'Senior (65+)'

    # Event type
    if 'Diet' in task_name:
        features['event_type'] = 'Diet'
    elif 'Exercise' in task_name:
        features['event_type'] = 'Exercise'
    elif 'Med' in task_name:
        features['event_type'] = 'Medication'

    # Context level
    if 'NoCtx' in task_name:
        features['context_type'] = 'No Context'
        features['context_level'] = 'None'
    elif 'DetailedEvent' in task_name:
        features['context_type'] = f"{features['event_type']} Context"
        features['context_level'] = 'Detailed Event'
    elif 'MediumEvent' in task_name:
        features['context_type'] = f"{features['event_type']} Context"
        features['context_level'] = 'Medium Event'
    elif 'Profile' in task_name:
        features['context_type'] = f"{features['event_type']} Context"
        features['context_level'] = 'Profile Only'

    return features


def compute_correlation_by_stratification(
    df: pd.DataFrame,
    metric: str,
    outcome: str = 'clarke_ab',
    stratify_by: str = 'context_type'
) -> Dict[str, Tuple[float, float, int]]:
    """
    Compute correlation between metric and outcome, stratified by feature.

    Returns:
        Dict mapping stratum → (correlation, std_err, n_samples)
    """
    # Add stratification column
    df['stratum'] = df['task'].apply(lambda x: extract_task_features(x)[stratify_by])

    results = {}

    # Overall correlation
    valid_overall = df[[metric, outcome]].dropna()
    if len(valid_overall) > 0:
        corr = valid_overall.corr().iloc[0, 1]
        # Standard error using Fisher z-transformation
        n = len(valid_overall)
        if n > 3:
            z = 0.5 * np.log((1 + corr) / (1 - corr))  # Fisher z
            se_z = 1 / np.sqrt(n - 3)
            results['Overall'] = (corr, se_z, n)
        else:
            results['Overall'] = (corr, 0.0, n)

    # Stratified correlations
    for stratum in df['stratum'].unique():
        if stratum == 'Unknown':
            continue

        stratum_data = df[df['stratum'] == stratum]
        valid = stratum_data[[metric, outcome]].dropna()

        if len(valid) > 3:
            corr = valid.corr().iloc[0, 1]
            n = len(valid)
            z = 0.5 * np.log((1 + corr) / (1 - corr))
            se_z = 1 / np.sqrt(n - 3)
            results[stratum] = (corr, se_z, n)

    return results


def create_correlation_table_latex(
    df: pd.DataFrame,
    stratifications: List[Tuple[str, str]],
    output_path: Path
):
    """
    Create LaTeX table showing metric correlations across stratifications.

    Args:
        df: DataFrame with all results
        stratifications: List of (stratify_by, label) tuples
        output_path: Where to save LaTeX file

    Table structure similar to CiK Table 1:
    - Rows: Metrics (CRPS, Glucose-RCRPS, Weighted CRPS, MAE, RMSE)
    - Columns: Overall + stratifications
    """
    metrics = [
        ('crps', 'Standard CRPS', 'Standard CRPS'),
        ('glucose_rcrps', 'Glucose-RCRPS', 'Glucose-RCRPS'),
        ('weighted_term', 'Weighted CRPS', 'Clarke-Weighted CRPS'),
        ('mae', 'MAE', 'MAE'),
        ('rmse', 'RMSE', 'RMSE'),
    ]

    # Collect all correlations
    all_correlations = {}
    for metric_col, _, metric_label in metrics:
        if metric_col not in df.columns:
            continue

        all_correlations[metric_label] = {}

        # Overall
        corr_results = compute_correlation_by_stratification(
            df, metric_col, 'clarke_ab', 'context_type'
        )
        if 'Overall' in corr_results:
            all_correlations[metric_label]['Overall'] = corr_results['Overall']

        # Each stratification
        for stratify_by, strat_label in stratifications:
            corr_results = compute_correlation_by_stratification(
                df, metric_col, 'clarke_ab', stratify_by
            )

            # Store all strata for this stratification
            for stratum, (corr, se, n) in corr_results.items():
                if stratum != 'Overall':
                    col_name = f"{strat_label}: {stratum}"
                    all_correlations[metric_label][col_name] = (corr, se, n)

    # Determine column order
    column_order = ['Overall']
    for stratify_by, strat_label in stratifications:
        # Get all unique strata
        strata = set()
        for metric_data in all_correlations.values():
            for col in metric_data.keys():
                if col.startswith(f"{strat_label}:"):
                    strata.add(col)
        column_order.extend(sorted(strata))

    # Start LaTeX
    latex = []
    latex.append("\\begin{table*}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Correlation with Clarke A+B (\\%) across different stratifications. "
                 "Values show Pearson correlation coefficient with standard error. "
                 "More negative values indicate stronger alignment with clinical safety. "
                 "Bold indicates best (most negative) correlation per column.}")
    latex.append("\\label{tab:correlation_analysis}")

    n_cols = 1 + len(column_order)  # metric name + columns
    latex.append(f"\\begin{{tabular}}{{l{'c' * len(column_order)}}}")
    latex.append("\\toprule")

    # Header row 1: Main categories
    header1 = "\\textsc{Metric}"
    current_category = ""
    category_spans = []
    col_idx = 0

    for col in column_order:
        if ':' in col:
            category = col.split(':')[0].strip()
            if category != current_category:
                if current_category:
                    category_spans.append((current_category, col_idx - len(category_spans)))
                current_category = category
        col_idx += 1

    # Simplified header for now
    header = "\\textsc{Metric} & " + " & ".join([
        f"\\textsc{{{col.split(': ')[1] if ': ' in col else col}}}"
        for col in column_order
    ]) + " \\\\"
    latex.append(header)
    latex.append("\\midrule")

    # Find best (most negative) value per column
    best_values = {}
    for col in column_order:
        values = []
        for metric_data in all_correlations.values():
            if col in metric_data:
                corr, _, _ = metric_data[col]
                values.append(corr)
        if values:
            best_values[col] = min(values)  # Most negative

    # Rows
    for metric_col, metric_short, metric_label in metrics:
        if metric_label not in all_correlations:
            continue

        row_values = [metric_label.replace('_', '\\_')]

        for col in column_order:
            if col in all_correlations[metric_label]:
                corr, se, n = all_correlations[metric_label][col]
                is_best = abs(corr - best_values[col]) < 1e-6

                # Format as correlation ± stderr
                val_str = f"{corr:.3f} \\pm {se:.3f}"
                if is_best:
                    val_str = f"\\textbf{{{val_str}}}"

                row_values.append(val_str)
            else:
                row_values.append("---")

        latex.append("  " + " & ".join(row_values) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table*}")

    # Save
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))

    print(f"Correlation table saved to: {output_path}")
    return '\n'.join(latex)


def create_summary_statistics_table(df: pd.DataFrame, output_path: Path):
    """
    Create summary table showing:
    - Overall correlation for each metric
    - Average correlation across all stratifications
    - Improvement vs CRPS baseline
    """
    metrics = ['crps', 'glucose_rcrps', 'weighted_term', 'mae', 'rmse']

    results = []
    for metric in metrics:
        if metric not in df.columns or 'clarke_ab' not in df.columns:
            continue

        valid = df[[metric, 'clarke_ab']].dropna()
        if len(valid) < 4:
            continue

        corr = valid.corr().iloc[0, 1]
        n = len(valid)

        results.append({
            'metric': metric,
            'correlation': corr,
            'n': n
        })

    results_df = pd.DataFrame(results)

    # Compute improvement vs CRPS
    if 'crps' in results_df['metric'].values:
        baseline_corr = results_df[results_df['metric'] == 'crps']['correlation'].iloc[0]
        results_df['improvement'] = (abs(results_df['correlation']) - abs(baseline_corr)) / abs(baseline_corr) * 100
    else:
        results_df['improvement'] = 0.0

    # Sort by correlation (most negative first)
    results_df = results_df.sort_values('correlation')

    # Create LaTeX
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Overall metric performance for clinical alignment}")
    latex.append("\\label{tab:metric_summary}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\toprule")
    latex.append("\\textsc{Metric} & \\textsc{Clarke Corr.} & \\textsc{N} & \\textsc{Improvement} \\\\")
    latex.append("\\midrule")

    metric_names = {
        'crps': 'Standard CRPS',
        'glucose_rcrps': 'Glucose-RCRPS',
        'weighted_term': 'Clarke-Weighted CRPS',
        'mae': 'MAE',
        'rmse': 'RMSE'
    }

    for _, row in results_df.iterrows():
        metric_name = metric_names.get(row['metric'], row['metric'])
        corr_str = f"{row['correlation']:.3f}"
        improvement_str = f"{row['improvement']:+.1f}\\%" if row['improvement'] != 0 else "baseline"

        # Bold best
        if row['metric'] in ['weighted_term', 'glucose_rcrps']:
            corr_str = f"\\textbf{{{corr_str}}}"

        latex.append(f"  {metric_name} & {corr_str} & {row['n']} & {improvement_str} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))

    print(f"Summary table saved to: {output_path}")
    return results_df


def main():
    """Generate correlation analysis tables."""
    print("=" * 80)
    print("CREATING CORRELATION ANALYSIS TABLES")
    print("=" * 80)

    # Setup paths
    repo_root = Path(__file__).resolve().parent.parent.parent
    csv_path = repo_root / 'evaluation' / 'results' / 'clinical_metrics_detailed.csv'
    output_dir = repo_root / 'evaluation' / 'results'

    # Load data
    print(f"\n1. Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    valid = df[df['glucose_rcrps'].notna()].copy()
    print(f"   Loaded {len(valid)} valid rows")

    # Define stratifications
    stratifications = [
        ('context_type', 'Context'),
        ('diabetes_type', 'Diabetes Type'),
        ('age_group', 'Age Group'),
        ('event_type', 'Event Type'),
    ]

    # Create correlation table
    print(f"\n2. Creating correlation analysis table...")
    corr_table_path = output_dir / 'correlation_analysis_table.tex'
    create_correlation_table_latex(valid, stratifications, corr_table_path)

    # Create summary table
    print(f"\n3. Creating summary statistics table...")
    summary_path = output_dir / 'metric_summary_table.tex'
    summary_df = create_summary_statistics_table(valid, summary_path)

    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nKey findings:")
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
