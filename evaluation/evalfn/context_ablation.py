"""
Context Ablation Analysis
==========================

Analyze CRPS across 4 context levels: NoCtx → Profile → MediumEvent → DetailedEvent

This addresses the key finding that "context hurts some models" by showing
the full gradient of context impact.

Usage:
    python evaluation/eval_fn/context_ablation.py

Output:
    - evaluation/results/context_ablation.csv
    - evaluation/results/context_ablation_summary.txt
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def read_crps_from_summary(summary_file: Path) -> Optional[float]:
    """Extract CRPS from data_summary.txt file."""
    if not summary_file.exists():
        return None

    try:
        with open(summary_file, 'r') as f:
            text = f.read()
            # Look for "crps: <value>" (with optional leading whitespace)
            match = re.search(r'^\s*crps:\s*([\d.]+)', text, re.MULTILINE)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"Warning: Failed to read {summary_file}: {e}")

    return None


def classify_context_level(task_name: str) -> Optional[str]:
    """Classify task by context level based on name."""
    if 'NoCtx' in task_name:
        return 'NoCtx'
    elif 'Profile' in task_name:
        return 'Profile'
    elif 'MediumEvent' in task_name:
        return 'MediumEvent'
    elif 'DetailedEvent' in task_name:
        return 'DetailedEvent'
    return None


def collect_context_results(result_dir: Path) -> pd.DataFrame:
    """
    Scan result directory and collect CRPS by (model, task, context_level).

    Returns DataFrame with columns:
        - model: str
        - task_base: str (task name without context suffix)
        - context_level: str (NoCtx/Profile/MediumEvent/DetailedEvent)
        - crps_mean: float (average over seeds)
        - crps_std: float
        - n_seeds: int
    """
    records = []

    for model_dir in sorted(result_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith('_'):
            continue

        # Normalize model name by stripping -nocontext/-context suffix
        model_name = model_dir.name
        model_base = re.sub(r'-(no)?context$', '', model_name)

        for task_dir in sorted(model_dir.iterdir()):
            if not task_dir.is_dir():
                continue

            task_name = task_dir.name
            context_level = classify_context_level(task_name)

            if context_level is None:
                continue  # Skip tasks without clear context level

            # Extract base task name (remove context suffix)
            task_base = re.sub(r'_(NoCtx|Profile|MediumEvent|DetailedEvent)$', '', task_name)

            # Collect CRPS from all seeds
            seed_crps = []
            for seed_dir in sorted(task_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue

                summary_file = seed_dir / 'data_summary.txt'
                crps = read_crps_from_summary(summary_file)
                if crps is not None:
                    seed_crps.append(crps)

            if seed_crps:
                records.append({
                    'model': model_base,  # Use normalized model name
                    'task_base': task_base,
                    'context_level': context_level,
                    'crps_mean': np.mean(seed_crps),
                    'crps_std': np.std(seed_crps),
                    'n_seeds': len(seed_crps)
                })

    return pd.DataFrame(records)


def compute_context_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Δ CRPS for each context level relative to NoCtx baseline.

    Returns DataFrame with columns:
        - model
        - task_base
        - crps_NoCtx
        - crps_Profile
        - crps_MediumEvent
        - crps_DetailedEvent
        - delta_Profile (Profile - NoCtx)
        - delta_MediumEvent
        - delta_DetailedEvent
    """
    # Pivot to wide format
    pivot = df.pivot_table(
        index=['model', 'task_base'],
        columns='context_level',
        values='crps_mean'
    ).reset_index()

    # Compute deltas (positive delta = worse with context)
    for level in ['Profile', 'MediumEvent', 'DetailedEvent']:
        if level in pivot.columns and 'NoCtx' in pivot.columns:
            pivot[f'delta_{level}'] = pivot[level] - pivot['NoCtx']

    return pivot


def aggregate_by_model(delta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate context deltas across all tasks for each model.

    Returns summary with mean/median delta per model.
    """
    summary = []

    for model in delta_df['model'].unique():
        model_data = delta_df[delta_df['model'] == model]

        record = {'model': model}

        # For each context level
        for level in ['Profile', 'MediumEvent', 'DetailedEvent']:
            delta_col = f'delta_{level}'
            if delta_col in model_data.columns:
                deltas = model_data[delta_col].dropna()
                if len(deltas) > 0:
                    record[f'{level}_mean_delta'] = deltas.mean()
                    record[f'{level}_median_delta'] = deltas.median()
                    record[f'{level}_n_tasks'] = len(deltas)
                    # % improvement (negative delta = improvement)
                    if 'NoCtx' in model_data.columns and model_data['NoCtx'].mean() != 0:
                        record[f'{level}_pct_improvement'] = -100 * deltas.mean() / model_data['NoCtx'].mean()

        if len(record) > 1:  # Has at least one context level
            summary.append(record)

    result_df = pd.DataFrame(summary)

    # Sort by the first available delta column
    for sort_col in ['DetailedEvent_mean_delta', 'MediumEvent_mean_delta', 'Profile_mean_delta']:
        if sort_col in result_df.columns:
            return result_df.sort_values(sort_col)

    return result_df


def identify_context_sensitive_models(summary_df: pd.DataFrame, threshold: float = 0.05) -> Dict[str, List[str]]:
    """
    Categorize models by context sensitivity.

    threshold: Absolute CRPS change threshold to be considered "sensitive"

    Returns dict with keys:
        - 'context_helps': models where context improves CRPS
        - 'context_hurts': models where context worsens CRPS
        - 'context_neutral': models where context has minimal impact
    """
    categorization = {
        'context_helps': [],
        'context_hurts': [],
        'context_neutral': []
    }

    for _, row in summary_df.iterrows():
        model = row['model']
        # Use DetailedEvent as the "full context" measure
        if 'DetailedEvent_mean_delta' in row and not pd.isna(row['DetailedEvent_mean_delta']):
            delta = row['DetailedEvent_mean_delta']
            if delta < -threshold:
                categorization['context_helps'].append(model)
            elif delta > threshold:
                categorization['context_hurts'].append(model)
            else:
                categorization['context_neutral'].append(model)

    return categorization


def main():
    """Run context ablation analysis."""
    print("=" * 80)
    print("CONTEXT ABLATION ANALYSIS")
    print("=" * 80)

    # Setup paths
    repo_root = Path(__file__).resolve().parent.parent.parent
    result_dir = repo_root / '_WorkSpace' / 'Result'
    output_dir = repo_root / 'evaluation' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    if not result_dir.exists():
        print(f"Error: Result directory not found: {result_dir}")
        return

    print(f"\n1. Scanning results from: {result_dir}")

    # Collect data
    df = collect_context_results(result_dir)

    if df.empty:
        print("Error: No context-level results found!")
        return

    print(f"   Found {len(df)} (model, task, context_level) combinations")
    print(f"   Models: {df['model'].nunique()}")
    print(f"   Base tasks: {df['task_base'].nunique()}")
    print(f"   Context levels: {df['context_level'].unique().tolist()}")

    # Compute deltas
    print("\n2. Computing context deltas (relative to NoCtx baseline)...")
    delta_df = compute_context_deltas(df)

    # Save detailed results
    detailed_file = output_dir / 'context_ablation_detailed.csv'
    delta_df.to_csv(detailed_file, index=False)
    print(f"   Saved detailed results: {detailed_file}")

    # Aggregate by model
    print("\n3. Aggregating by model...")
    summary_df = aggregate_by_model(delta_df)

    summary_file = output_dir / 'context_ablation_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"   Saved summary: {summary_file}")

    # Categorize models
    print("\n4. Categorizing models by context sensitivity...")
    categories = identify_context_sensitive_models(summary_df, threshold=0.05)

    # Print summary report
    report_file = output_dir / 'context_ablation_report.txt'
    with open(report_file, 'w') as f:
        f.write("CONTEXT ABLATION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total models analyzed: {len(summary_df)}\n")
        f.write(f"Models with context data: {len(summary_df)}\n\n")

        # Find which delta column is available
        delta_col = None
        for col in ['DetailedEvent_mean_delta', 'MediumEvent_mean_delta', 'Profile_mean_delta']:
            if col in summary_df.columns:
                delta_col = col
                break

        if delta_col is None:
            f.write("No context level data found for comparison!\n")
        else:
            level_name = delta_col.replace('_mean_delta', '')
            pct_col = f'{level_name}_pct_improvement'

            f.write(f"TOP 10 MODELS BY CONTEXT BENEFIT ({level_name})\n")
            f.write("-" * 80 + "\n")
            top10 = summary_df.nsmallest(10, delta_col) if len(summary_df) >= 10 else summary_df.sort_values(delta_col)
            for _, row in top10.iterrows():
                if not pd.isna(row[delta_col]):
                    pct_str = f"({row[pct_col]:+.1f}%)" if pct_col in row and not pd.isna(row[pct_col]) else ""
                    f.write(f"{row['model']:50s}  Δ CRPS: {row[delta_col]:+.4f}  {pct_str}\n")

            f.write(f"\n\nTOP 10 MODELS WHERE CONTEXT HURTS ({level_name})\n")
            f.write("-" * 80 + "\n")
            bottom10 = summary_df.nlargest(10, delta_col) if len(summary_df) >= 10 else summary_df.sort_values(delta_col, ascending=False)
            for _, row in bottom10.iterrows():
                if not pd.isna(row[delta_col]):
                    pct_str = f"({row[pct_col]:+.1f}%)" if pct_col in row and not pd.isna(row[pct_col]) else ""
                    f.write(f"{row['model']:50s}  Δ CRPS: {row[delta_col]:+.4f}  {pct_str}\n")

        f.write("\n\nMODEL CATEGORIZATION (threshold = 0.05 CRPS)\n")
        f.write("-" * 80 + "\n")
        f.write(f"\nContext HELPS ({len(categories['context_helps'])} models):\n")
        for model in categories['context_helps']:
            f.write(f"  - {model}\n")

        f.write(f"\nContext HURTS ({len(categories['context_hurts'])} models):\n")
        for model in categories['context_hurts']:
            f.write(f"  - {model}\n")

        f.write(f"\nContext NEUTRAL ({len(categories['context_neutral'])} models):\n")
        for model in categories['context_neutral']:
            f.write(f"  - {model}\n")

    print(f"   Saved report: {report_file}")

    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print(f"\nContext HELPS: {len(categories['context_helps'])} models")
    print(f"Context HURTS: {len(categories['context_hurts'])} models")
    print(f"Context NEUTRAL: {len(categories['context_neutral'])} models")

    # Find available delta column
    delta_col = None
    for col in ['DetailedEvent_mean_delta', 'MediumEvent_mean_delta', 'Profile_mean_delta']:
        if col in summary_df.columns:
            delta_col = col
            break

    if delta_col:
        level_name = delta_col.replace('_mean_delta', '')
        pct_col = f'{level_name}_pct_improvement'

        print(f"\nTop 3 models that benefit from context ({level_name}):")
        for i, (_, row) in enumerate(summary_df.nsmallest(3, delta_col).iterrows(), 1):
            if pct_col in row and not pd.isna(row[pct_col]):
                print(f"  {i}. {row['model']}: {row[pct_col]:+.1f}% improvement")

        print(f"\nTop 3 models where context hurts ({level_name}):")
        for i, (_, row) in enumerate(summary_df.nlargest(3, delta_col).iterrows(), 1):
            if pct_col in row and not pd.isna(row[pct_col]):
                print(f"  {i}. {row['model']}: {row[pct_col]:.1f}% worse")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - {detailed_file}")
    print(f"  - {summary_file}")
    print(f"  - {report_file}")


if __name__ == '__main__':
    main()
