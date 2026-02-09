"""
Demographic Subgroup Analysis
==============================

Analyze model performance across demographic groups:
- Diabetes type: D1 (Type 1) vs D2 (Type 2)
- Age groups: 18, 40, 65
- Event types: Diet, Exercise, NoEvent

Computes fairness metrics (disparity, Gini coefficient).

Usage:
    python evaluation/eval_fn/demographic_analysis.py

Output:
    - evaluation/results/demographic_analysis.csv
    - evaluation/results/demographic_fairness.csv
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


def extract_demographics(task_name: str) -> dict:
    """Extract demographic info from task name."""
    demo = {
        'diabetes_type': None,
        'age': None,
        'event_type': None
    }

    # Diabetes type
    if '_D1_' in task_name:
        demo['diabetes_type'] = 'D1'
    elif '_D2_' in task_name:
        demo['diabetes_type'] = 'D2'

    # Age
    age_match = re.search(r'Age(\d+)', task_name)
    if age_match:
        demo['age'] = int(age_match.group(1))

    # Event type
    if 'Diet' in task_name:
        demo['event_type'] = 'Diet'
    elif 'Exercise' in task_name:
        demo['event_type'] = 'Exercise'
    elif 'NoEvent' in task_name:
        demo['event_type'] = 'NoEvent'

    return demo


def load_aggregated_results(result_dir: Path) -> pd.DataFrame:
    """Load the aggregated_results.csv if it exists, otherwise create it."""
    agg_file = result_dir / 'aggregated_results.csv'

    if agg_file.exists():
        return pd.read_csv(agg_file)

    # If not exists, create a minimal version by scanning
    print("  Note: aggregated_results.csv not found, scanning directories...")
    records = []

    for model_dir in sorted(result_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith('_'):
            continue

        model_name = model_dir.name

        for task_dir in sorted(model_dir.iterdir()):
            if not task_dir.is_dir():
                continue

            task_name = task_dir.name

            # Collect CRPS from seeds
            seed_crps = []
            for seed_dir in sorted(task_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue

                summary_file = seed_dir / 'data_summary.txt'
                if summary_file.exists():
                    try:
                        with open(summary_file, 'r') as f:
                            text = f.read()
                            match = re.search(r'^crps:\s*([\d.]+)', text, re.MULTILINE)
                            if match:
                                seed_crps.append(float(match.group(1)))
                    except:
                        pass

            if seed_crps:
                records.append({
                    'model': model_name,
                    'task': task_name,
                    'crps_mean': np.mean(seed_crps),
                    'crps_std': np.std(seed_crps),
                    'n_seeds': len(seed_crps)
                })

    return pd.DataFrame(records)


def compute_gini_coefficient(values: np.ndarray) -> float:
    """
    Compute Gini coefficient for disparity measurement.

    Returns value in [0, 1] where 0 = perfect equality, 1 = maximum inequality.
    """
    if len(values) == 0:
        return 0.0

    sorted_values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_values)
    return (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n


def analyze_fairness(df: pd.DataFrame, group_col: str) -> dict:
    """
    Compute fairness metrics for a grouping variable.

    Metrics:
    - max_error: highest group CRPS
    - min_error: lowest group CRPS
    - spread: max - min
    - gini: Gini coefficient of group errors
    """
    group_errors = df.groupby(group_col)['crps_mean'].mean()

    if len(group_errors) == 0:
        return {}

    return {
        'max_error': group_errors.max(),
        'min_error': group_errors.min(),
        'spread': group_errors.max() - group_errors.min(),
        'gini': compute_gini_coefficient(group_errors.values),
        'n_groups': len(group_errors)
    }


def main():
    """Run demographic analysis."""
    print("=" * 80)
    print("DEMOGRAPHIC SUBGROUP ANALYSIS")
    print("=" * 80)

    # Setup paths
    repo_root = Path(__file__).resolve().parent.parent.parent
    result_dir = repo_root / '_WorkSpace' / 'Result'
    output_dir = repo_root / 'evaluation' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    if not result_dir.exists():
        print(f"Error: Result directory not found: {result_dir}")
        return

    print(f"\n1. Loading results from: {result_dir}")

    # Load data
    df = load_aggregated_results(result_dir)

    if df.empty:
        print("Error: No results found!")
        return

    print(f"   Loaded {len(df)} model-task combinations")

    # Extract demographics
    print("\n2. Extracting demographic information...")
    demo_df = df.copy()
    demo_info = demo_df['task'].apply(extract_demographics)
    demo_df['diabetes_type'] = demo_info.apply(lambda x: x['diabetes_type'])
    demo_df['age'] = demo_info.apply(lambda x: x['age'])
    demo_df['event_type'] = demo_info.apply(lambda x: x['event_type'])

    # Overall statistics by demographic group
    print("\n3. Computing group statistics...")

    # Diabetes type
    diabetes_stats = demo_df.groupby('diabetes_type')['crps_mean'].agg(['mean', 'std', 'count']).reset_index()
    diabetes_stats.columns = ['diabetes_type', 'crps_mean', 'crps_std', 'n_tasks']

    # Age
    age_stats = demo_df.groupby('age')['crps_mean'].agg(['mean', 'std', 'count']).reset_index()
    age_stats.columns = ['age', 'crps_mean', 'crps_std', 'n_tasks']

    # Event type
    event_stats = demo_df.groupby('event_type')['crps_mean'].agg(['mean', 'std', 'count']).reset_index()
    event_stats.columns = ['event_type', 'crps_mean', 'crps_std', 'n_tasks']

    # Save demographic summaries
    demo_file = output_dir / 'demographic_summary.csv'
    with open(demo_file, 'w') as f:
        f.write("# DIABETES TYPE\n")
        diabetes_stats.to_csv(f, index=False)
        f.write("\n# AGE\n")
        age_stats.to_csv(f, index=False)
        f.write("\n# EVENT TYPE\n")
        event_stats.to_csv(f, index=False)

    print(f"   Saved demographic summary: {demo_file}")

    # Fairness analysis per model
    print("\n4. Computing fairness metrics per model...")
    fairness_records = []

    for model in sorted(demo_df['model'].unique()):
        model_data = demo_df[demo_df['model'] == model]

        record = {'model': model}

        # Fairness by diabetes type
        if model_data['diabetes_type'].notna().any():
            diabetes_fairness = analyze_fairness(model_data, 'diabetes_type')
            for k, v in diabetes_fairness.items():
                record[f'diabetes_{k}'] = v

        # Fairness by age
        if model_data['age'].notna().any():
            age_fairness = analyze_fairness(model_data, 'age')
            for k, v in age_fairness.items():
                record[f'age_{k}'] = v

        # Fairness by event type
        if model_data['event_type'].notna().any():
            event_fairness = analyze_fairness(model_data, 'event_type')
            for k, v in event_fairness.items():
                record[f'event_{k}'] = v

        fairness_records.append(record)

    fairness_df = pd.DataFrame(fairness_records)

    fairness_file = output_dir / 'demographic_fairness.csv'
    fairness_df.to_csv(fairness_file, index=False)
    print(f"   Saved fairness metrics: {fairness_file}")

    # Create report
    report_file = output_dir / 'demographic_report.txt'
    with open(report_file, 'w') as f:
        f.write("DEMOGRAPHIC ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("OVERALL GROUP STATISTICS\n")
        f.write("-" * 80 + "\n\n")

        f.write("Diabetes Type:\n")
        f.write(diabetes_stats.to_string(index=False))
        f.write("\n\n")

        f.write("Age Groups:\n")
        f.write(age_stats.to_string(index=False))
        f.write("\n\n")

        f.write("Event Types:\n")
        f.write(event_stats.to_string(index=False))
        f.write("\n\n")

        f.write("TOP 5 MOST FAIR MODELS (lowest age spread)\n")
        f.write("-" * 80 + "\n")
        if 'age_spread' in fairness_df.columns:
            top5_fair = fairness_df.nsmallest(5, 'age_spread')
            for _, row in top5_fair.iterrows():
                f.write(f"{row['model']:50s}  Age Spread: {row['age_spread']:.4f}\n")

        f.write("\n\nTOP 5 MOST UNFAIR MODELS (highest age spread)\n")
        f.write("-" * 80 + "\n")
        if 'age_spread' in fairness_df.columns:
            top5_unfair = fairness_df.nlargest(5, 'age_spread')
            for _, row in top5_unfair.iterrows():
                f.write(f"{row['model']:50s}  Age Spread: {row['age_spread']:.4f}\n")

    print(f"   Saved report: {report_file}")

    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    print("\nDiabetes Type:")
    for _, row in diabetes_stats.iterrows():
        print(f"  {row['diabetes_type']}: {row['crps_mean']:.4f} ± {row['crps_std']:.4f}")

    print("\nAge Groups (increasing difficulty):")
    for _, row in age_stats.sort_values('age').iterrows():
        print(f"  Age {int(row['age'])}: {row['crps_mean']:.4f} ± {row['crps_std']:.4f}")

    print("\nEvent Types:")
    for _, row in event_stats.iterrows():
        print(f"  {row['event_type']}: {row['crps_mean']:.4f} ± {row['crps_std']:.4f}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - {demo_file}")
    print(f"  - {fairness_file}")
    print(f"  - {report_file}")


if __name__ == '__main__':
    main()
