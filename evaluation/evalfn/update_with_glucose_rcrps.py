"""
Update clinical_metrics_detailed.csv with Glucose-RCRPS values
================================================================

This script:
1. Reads existing clinical_metrics_detailed.csv
2. For each (model, task, seed), loads complete_data.pkl
3. Computes Glucose-RCRPS using compute_glucose_rcrps.py
4. Adds glucose_rcrps column to CSV
5. Saves updated CSV with component breakdown

Usage:
    python 0--evaluation/evalfn/update_with_glucose_rcrps.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
repo_root = Path(__file__).resolve().parent.parent.parent

import numpy as np
import pandas as pd
from evalfn.measurements import load_complete_data
from evalfn.compute_glucose_rcrps import compute_glucose_rcrps
from evalfn.util_crps import crps


def extract_prediction_data(pkl_path: Path, target_col: str = None, target_index: int = -1):
    """
    Extract all necessary data from complete_data.pkl for Glucose-RCRPS computation.

    Args:
        pkl_path: Path to complete_data.pkl
        target_col: Optional column name for ground truth
        target_index: Index of target variable (default: -1, last column)

    Returns:
        dict with keys:
        - actual_values: Ground truth trajectory (n_timesteps,)
        - predicted_values: Point forecast (median of samples) (n_timesteps,)
        - predicted_samples: Full distribution (n_samples, n_timesteps)
        - crps_values: Per-timestep CRPS (n_timesteps,)
        Returns None if extraction fails.
    """
    try:
        data = load_complete_data(pkl_path)

        # Extract predictions and ground truth
        samples = np.asarray(data['predictions']['samples'])  # (n_samples, horizon, n_vars)
        future_time = data['input_data']['future_time']

        # Get ground truth (n_timesteps,)
        if target_col and hasattr(future_time, target_col):
            y_true = np.asarray(future_time[target_col], dtype=float)
        else:
            y_true = np.asarray(future_time, dtype=float)
            if y_true.ndim == 2:
                y_true = y_true[:, target_index]

        # Extract samples for target variable
        # samples shape: (n_samples, horizon, n_vars) → (n_samples, horizon)
        if samples.ndim == 3:
            samples_2d = samples[:, :, target_index]  # (n_samples, horizon)
        else:
            samples_2d = samples  # Already (n_samples, horizon)

        # Align lengths (defensive programming)
        min_len = min(samples_2d.shape[1], len(y_true))
        samples_2d = samples_2d[:, :min_len]
        y_true = y_true[:min_len]

        # Compute point forecast (median of samples)
        # Shape: (n_timesteps,)
        y_pred = np.median(samples_2d, axis=0)

        # Compute per-timestep CRPS
        # crps() expects: target shape (n_timesteps,), samples shape (n_samples, n_timesteps)
        # Returns: array of shape (n_timesteps,)
        crps_values = crps(y_true, samples_2d)

        return {
            'actual_values': y_true,
            'predicted_values': y_pred,
            'predicted_samples': samples_2d,
            'crps_values': crps_values
        }

    except Exception as e:
        print(f"Warning: Failed to extract data from {pkl_path}: {e}")
        return None


def compute_glucose_rcrps_for_row(result_dir: Path, model: str, task: str, seed):
    """
    Compute Glucose-RCRPS for a single (model, task, seed) combination.

    Args:
        result_dir: Root directory containing results (_WorkSpace/Result)
        model: Model name
        task: Task name
        seed: Seed ID (int or str)

    Returns:
        dict with glucose_rcrps results, or None if computation fails
    """
    # Build path to complete_data.pkl
    # Structure: result_dir / model / task / seed / complete_data.pkl
    # Convert seed to string in case it's read as int from CSV
    pkl_path = result_dir / model / task / str(seed) / 'complete_data.pkl'

    if not pkl_path.exists():
        print(f"Warning: {pkl_path} does not exist")
        return None

    # Extract prediction data
    pred_data = extract_prediction_data(pkl_path)
    if pred_data is None:
        return None

    # Compute Glucose-RCRPS
    try:
        results = compute_glucose_rcrps(
            crps_values=pred_data['crps_values'],
            actual_values=pred_data['actual_values'],
            predicted_values=pred_data['predicted_values'],
            predicted_samples=pred_data['predicted_samples']
        )
        return results
    except Exception as e:
        print(f"Warning: Failed to compute Glucose-RCRPS for {model}/{task}/{seed}: {e}")
        return None


def main():
    """Update clinical_metrics_detailed.csv with Glucose-RCRPS values."""
    print("=" * 80)
    print("GLUCOSE-RCRPS UPDATE PIPELINE")
    print("=" * 80)

    # Setup paths
    repo_root = Path(__file__).resolve().parent.parent.parent
    result_dir = repo_root / '_WorkSpace' / 'Result'
    csv_path = repo_root / 'evaluation' / 'results' / 'clinical_metrics_detailed.csv'

    if not result_dir.exists():
        print(f"Error: Result directory not found: {result_dir}")
        return

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return

    # Load existing CSV
    print(f"\n1. Loading existing CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Found {len(df)} rows")

    # Compute Glucose-RCRPS for each row
    print(f"\n2. Computing Glucose-RCRPS for each row...")
    print("   This may take several minutes...")

    glucose_rcrps_values = []
    clarke_weighted_crps_values = []
    constraint_crps_values = []
    weighted_term_values = []
    penalty_term_values = []

    for idx, row in df.iterrows():
        model = row['model']
        task = row['task']
        seed = row['seed']

        if idx % 10 == 0:
            print(f"   Processing row {idx + 1}/{len(df)}: {model}/{task}/{seed}")

        # Compute Glucose-RCRPS
        results = compute_glucose_rcrps_for_row(result_dir, model, task, seed)

        if results is not None:
            glucose_rcrps_values.append(results['glucose_rcrps'])
            clarke_weighted_crps_values.append(results['clarke_weighted_crps'])
            constraint_crps_values.append(results['constraint_crps'])
            weighted_term_values.append(results['weighted_term'])
            penalty_term_values.append(results['penalty_term'])
        else:
            # Fill with NaN if computation failed
            glucose_rcrps_values.append(np.nan)
            clarke_weighted_crps_values.append(np.nan)
            constraint_crps_values.append(np.nan)
            weighted_term_values.append(np.nan)
            penalty_term_values.append(np.nan)

    # Add columns to DataFrame
    print(f"\n3. Adding Glucose-RCRPS columns to CSV...")
    df['glucose_rcrps'] = glucose_rcrps_values
    df['clarke_weighted_crps'] = clarke_weighted_crps_values
    df['constraint_crps'] = constraint_crps_values
    df['weighted_term'] = weighted_term_values
    df['penalty_term'] = penalty_term_values

    # Save updated CSV
    output_path = repo_root / 'evaluation' / 'results' / 'clinical_metrics_detailed.csv'
    df.to_csv(output_path, index=False)
    print(f"   Saved updated CSV: {output_path}")

    # Print summary statistics
    print(f"\n4. Summary Statistics:")
    print(f"   Glucose-RCRPS computed for {df['glucose_rcrps'].notna().sum()} / {len(df)} rows")
    print(f"   Mean Glucose-RCRPS: {df['glucose_rcrps'].mean():.4f}")
    print(f"   Std Glucose-RCRPS: {df['glucose_rcrps'].std():.4f}")

    # Correlation analysis
    if 'crps' in df.columns and df['crps'].notna().sum() > 0:
        corr_crps = df[['crps', 'glucose_rcrps']].corr().iloc[0, 1]
        print(f"\n5. Correlation Analysis:")
        print(f"   CRPS vs Glucose-RCRPS: {corr_crps:.3f}")

    if 'clarke_ab' in df.columns and df['clarke_ab'].notna().sum() > 0:
        corr_clarke = df[['clarke_ab', 'glucose_rcrps']].corr().iloc[0, 1]
        print(f"   Clarke A+B vs Glucose-RCRPS: {corr_clarke:.3f}")
        print(f"   (Target: < -0.5, previous CRPS baseline: -0.378)")

    print("\n" + "=" * 80)
    print("UPDATE COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
