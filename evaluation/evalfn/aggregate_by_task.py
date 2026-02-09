#!/usr/bin/env python3
"""
Aggregate Clinical Metrics by Model and Task
=============================================

Create a table grouped by (model, task), averaging across all seeds/instances.

Input:  evaluation/results/clinical_metrics_detailed.csv
Output: evaluation/results/clinical_metrics_by_task.csv

Usage:
    python evaluation/eval_fn/aggregate_by_task.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))


def aggregate_by_model_task(detailed_csv: Path, output_csv: Path):
    """
    Aggregate clinical metrics by (model, task), averaging across seeds.

    Input columns:
        model, task, seed, crps, clarke_a, clarke_b, clarke_ab,
        clarke_c, clarke_d, clarke_e, mae, rmse

    Output columns:
        model, task, n_instances,
        crps_mean, crps_std,
        mae_mean, mae_std,
        rmse_mean, rmse_std,
        clarke_a_mean, clarke_a_std,
        clarke_b_mean, clarke_b_std,
        clarke_ab_mean, clarke_ab_std,
        clarke_c_mean, clarke_c_std,
        clarke_d_mean, clarke_d_std,
        clarke_e_mean, clarke_e_std
    """

    print("=" * 80)
    print("AGGREGATE CLINICAL METRICS BY MODEL AND TASK")
    print("=" * 80)
    print()

    # Read detailed data
    print(f"Reading: {detailed_csv}")
    df = pd.read_csv(detailed_csv)
    print(f"  Loaded {len(df)} rows")
    print()

    # Define metrics to aggregate
    metrics = ['crps', 'mae', 'rmse',
               'clarke_a', 'clarke_b', 'clarke_ab',
               'clarke_c', 'clarke_d', 'clarke_e']

    # Group by (model, task)
    print("Aggregating by (model, task)...")
    grouped = df.groupby(['model', 'task'])

    # Compute statistics
    agg_dict = {}

    # Count instances
    agg_dict['n_instances'] = grouped['seed'].count()

    # For each metric, compute mean and std
    for metric in metrics:
        if metric in df.columns:
            agg_dict[f'{metric}_mean'] = grouped[metric].mean()
            agg_dict[f'{metric}_std'] = grouped[metric].std()

    # Create aggregated dataframe
    result = pd.DataFrame(agg_dict).reset_index()

    # Sort by model, then by task
    result = result.sort_values(['model', 'task'])

    # Save
    print(f"Saving: {output_csv}")
    result.to_csv(output_csv, index=False)
    print(f"  Saved {len(result)} rows")
    print()

    # Summary statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total model-task pairs: {len(result)}")
    print(f"Unique models: {result['model'].nunique()}")
    print(f"Unique tasks: {result['task'].nunique()}")
    print()

    # Show distribution of instances per (model, task)
    print("Instances per (model, task) distribution:")
    print(result['n_instances'].describe())
    print()

    # Show sample rows
    print("Sample rows (first 5):")
    print(result.head()[['model', 'task', 'n_instances', 'crps_mean', 'mae_mean', 'clarke_ab_mean']])
    print()

    print("=" * 80)
    print("DONE")
    print("=" * 80)


def main():
    # Setup paths
    repo_root = Path(__file__).resolve().parent.parent.parent
    results_dir = repo_root / 'evaluation' / 'results'

    detailed_csv = results_dir / 'clinical_metrics_detailed.csv'
    output_csv = results_dir / 'clinical_metrics_by_task.csv'

    if not detailed_csv.exists():
        print(f"ERROR: Input file not found: {detailed_csv}")
        print("Please run: python evaluation/eval_fn/clinical_metrics.py")
        sys.exit(1)

    aggregate_by_model_task(detailed_csv, output_csv)


if __name__ == '__main__':
    main()
