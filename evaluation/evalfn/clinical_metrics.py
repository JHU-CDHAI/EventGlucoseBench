"""
Clinical Metrics Computation
=============================

Compute CRPS variants, Clarke Error Grid, MAE, and RMSE from saved prediction samples.

All metrics are computed directly from complete_data.pkl files:

CRPS Progression (4 variants):
- crps: Standard CRPS (baseline)
- t_crps: Temporal-weighted CRPS (emphasizes long-term predictions)
- c_crps: Clarke-weighted CRPS (distribution-aware, emphasizes clinical danger)
- ct_crps: Clarke-Temporal-weighted CRPS (both temporal + Clarke)

Clarke Error Grid Analysis:
- Individual zones: clarke_a, clarke_b, clarke_c, clarke_d, clarke_e (%)
- clarke_ab: % clinically acceptable (A+B)
- clarke_cde: % clinically problematic (C+D+E)

Point Forecast Metrics:
- mae: Mean Absolute Error (mg/dL)
- rmse: Root Mean Square Error (mg/dL)

Usage:
    python 0--evaluation/evalfn/clinical_metrics.py

Output:
    - evaluation/results/clinical_metrics.csv
    - evaluation/results/clinical_metrics_summary.txt
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
repo_root = Path(__file__).resolve().parent.parent.parent

import numpy as np
import pandas as pd
from evalfn.measurements import load_complete_data, n_samples_from_complete_data
from evalfn.util_crps import crps as compute_crps


def clarke_error_grid_zone(y_true, y_pred):
    """
    Classify a single (reference, prediction) pair into Clarke Error Grid zones.

    Returns zone: 'A', 'B', 'C', 'D', or 'E'

    Zone A: Clinically accurate
    Zone B: Benign errors (acceptable)
    Zone C: Overcorrection
    Zone D: Failure to detect
    Zone E: Erroneous treatment

    Based on Clarke et al. "Evaluating Clinical Accuracy of Systems for
    Self-Monitoring of Blood Glucose" Diabetes Care 1987.
    """
    # Handle edge cases
    if y_true < 0 or y_pred < 0:
        return 'E'

    # Zone A: Clinically accurate (within 20% or both in hypoglycemic range)
    if y_true <= 70 and y_pred <= 70:
        return 'A'  # Both hypoglycemic

    if y_true >= 180:
        if 0.75 * y_true <= y_pred <= 1.25 * y_true:
            return 'A'
    else:  # 70 < y_true < 180
        if 0.8 * y_true <= y_pred <= 1.2 * y_true:
            return 'A'

    # Zone B: Benign errors
    if (y_true >= 70 and y_pred >= 70) or (y_true <= 70 and y_pred <= 70):
        # Both in same general range, but outside Zone A
        if y_true < 70:
            # Both hypoglycemic
            return 'B'
        elif y_true > 180 and y_pred > 180:
            # Both hyperglycemic
            return 'B'
        elif 70 <= y_true <= 180 and 70 <= y_pred <= 180:
            # Both in target range
            return 'B'
        elif abs(y_pred - y_true) <= 50:
            # Small absolute error
            return 'B'

    # Zone C: Overcorrection (predict high when true is low, or vice versa)
    if (y_true < 70 and y_pred > 180) or (y_true > 180 and y_pred < 70):
        return 'C'

    # Zone D: Failure to detect (miss hypo/hyper)
    if (y_true < 70 and y_pred > 70) or (y_true > 180 and 70 < y_pred < 180):
        return 'D'
    if (y_true > 180 and y_pred < 180) or (y_true < 70 and 70 < y_pred < 180):
        return 'D'

    # Zone E: Erroneous treatment (opposite error)
    return 'E'


def compute_clarke_from_pickle(pkl_path: Path, target_col: str = None, target_index: int = -1):
    """
    Compute Clarke Error Grid percentages from complete_data.pkl.

    Uses sample median as point forecast.

    Returns dict with:
        - clarke_a: % in Zone A (clinically accurate)
        - clarke_b: % in Zone B (benign errors)
        - clarke_c: % in Zone C (overcorrection)
        - clarke_d: % in Zone D (failure to detect)
        - clarke_e: % in Zone E (erroneous treatment)
        - clarke_ab: % in Zones A+B (clinically acceptable)
        - clarke_cde: % in Zones C+D+E (clinically problematic)
    """
    try:
        data = load_complete_data(pkl_path)

        # Extract predictions and ground truth
        samples = np.asarray(data['predictions']['samples'])
        future_time = data['input_data']['future_time']

        # Get ground truth
        if target_col and hasattr(future_time, target_col):
            y_true = np.asarray(future_time[target_col], dtype=float)
        else:
            y_true = np.asarray(future_time, dtype=float)
            if y_true.ndim == 2:
                y_true = y_true[:, target_index]

        # Get point forecast (median over samples)
        if samples.ndim == 3:
            samples_2d = samples[:, :, target_index]
        else:
            samples_2d = samples

        y_pred = np.median(samples_2d, axis=0)

        # Align lengths
        min_len = min(len(y_pred), len(y_true))
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]

        # Classify each point
        zones = [clarke_error_grid_zone(yt, yp) for yt, yp in zip(y_true, y_pred)]

        # Compute percentages
        total = len(zones)
        zone_counts = {z: zones.count(z) for z in ['A', 'B', 'C', 'D', 'E']}

        return {
            'clarke_a': zone_counts['A'] / total * 100,
            'clarke_b': zone_counts['B'] / total * 100,
            'clarke_c': zone_counts['C'] / total * 100,
            'clarke_d': zone_counts['D'] / total * 100,
            'clarke_e': zone_counts['E'] / total * 100,
            'clarke_ab': (zone_counts['A'] + zone_counts['B']) / total * 100,
            'clarke_cde': (zone_counts['C'] + zone_counts['D'] + zone_counts['E']) / total * 100,
            'n_timesteps': int(total)
        }

    except Exception as e:
        print(f"Warning: Failed to compute Clarke for {pkl_path}: {e}")
        return None


def compute_mae_rmse_from_pickle(pkl_path: Path, target_col: str = None, target_index: int = -1):
    """
    Compute MAE and RMSE from complete_data.pkl.

    Uses sample median as point forecast.
    """
    try:
        data = load_complete_data(pkl_path)

        # Extract predictions and ground truth
        samples = np.asarray(data['predictions']['samples'])  # (n_samples, horizon, n_vars)
        future_time = data['input_data']['future_time']

        # Get ground truth
        if target_col and hasattr(future_time, target_col):
            y_true = np.asarray(future_time[target_col], dtype=float)
        else:
            y_true = np.asarray(future_time, dtype=float)
            if y_true.ndim == 2:
                y_true = y_true[:, target_index]

        # Get point forecast (median over samples)
        if samples.ndim == 3:
            samples_2d = samples[:, :, target_index]  # (n_samples, horizon)
        else:
            samples_2d = samples  # (n_samples, horizon)

        y_pred = np.median(samples_2d, axis=0)  # (horizon,)

        # Align lengths
        min_len = min(len(y_pred), len(y_true))
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]

        # Compute metrics
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'n_timesteps': int(min_len)
        }

    except Exception as e:
        print(f"Warning: Failed to compute MAE/RMSE for {pkl_path}: {e}")
        return None


def compute_crps_from_pickle(pkl_path: Path, target_col: str = None, target_index: int = -1):
    """
    Compute CRPS from complete_data.pkl samples.

    Uses the empirical CRPS formula from util_crps.py.
    Computes CRPS per timestep, then returns the mean.

    Returns dict with:
        - crps: Mean CRPS across all timesteps (mg/dL)
        - n_timesteps: Number of timesteps
    """
    try:
        data = load_complete_data(pkl_path)

        # Extract predictions and ground truth
        samples = np.asarray(data['predictions']['samples'])
        future_time = data['input_data']['future_time']

        # Get ground truth
        if target_col and hasattr(future_time, target_col):
            y_true = np.asarray(future_time[target_col], dtype=float)
        else:
            y_true = np.asarray(future_time, dtype=float)
            if y_true.ndim == 2:
                y_true = y_true[:, target_index]

        # Get samples in correct shape (S, T) where S=num_samples, T=horizon
        if samples.ndim == 3:
            samples_2d = samples[:, :, target_index]  # (n_samples, horizon)
        else:
            samples_2d = samples  # (n_samples, horizon)

        # Align lengths
        min_len = min(samples_2d.shape[1], len(y_true))
        samples_2d = samples_2d[:, :min_len]
        y_true = y_true[:min_len]

        # Compute CRPS per timestep using util_crps.py
        # Input: target (T,), samples (S, T) → Output: CRPS per timestep (T,)
        crps_per_timestep = compute_crps(y_true, samples_2d)

        # Average across timesteps to get mean CRPS
        mean_crps = float(np.mean(crps_per_timestep))

        return {
            'crps': mean_crps,
            'n_timesteps': int(min_len)
        }

    except Exception as e:
        print(f"Warning: Failed to compute CRPS for {pkl_path}: {e}")
        return None


def compute_time_weighted_crps_from_pickle(
    pkl_path: Path,
    weight_scheme: str = 'clinical_horizons',
    target_col: str = None,
    target_index: int = -1
):
    """
    Compute temporal-weighted CRPS (t_crps) from complete_data.pkl samples.

    Weights later timesteps more heavily to emphasize long-term prediction accuracy.
    This is clinical-temporal aware: longer prediction horizons are harder and more
    valuable to predict accurately.

    Uses normalized weights (sum=1) to keep units in mg/dL and remain comparable
    to standard CRPS.

    Weight schemes:
        - 'clinical_horizons' (default): Based on clinical relevance
            * 0-15 min (t=1-3): weight = 1.0 (baseline)
            * 15-30 min (t=4-6): weight = 1.5
            * 30-60 min (t=7-12): weight = 2.0
            * 60-120 min (t=13-24): weight = 3.0 (most important)

        - 'hour_based': Simple two-tier weighting
            * First hour: weight = 1.0
            * Second hour: weight = 2.0

        - 'linear': Linear increase with timestep
            * weight_t = t (1, 2, 3, ..., 24)

    Args:
        pkl_path: Path to complete_data.pkl
        weight_scheme: Weighting strategy ('clinical_horizons', 'hour_based', 'linear')
        target_col: Optional column name for target
        target_index: Index to extract from multidimensional targets

    Returns dict with:
        - t_crps: Temporal-weighted CRPS (mg/dL)
        - crps_standard: Standard mean CRPS for comparison (mg/dL)
        - n_timesteps: Number of timesteps
    """
    try:
        data = load_complete_data(pkl_path)

        # Extract predictions and ground truth
        samples = np.asarray(data['predictions']['samples'])
        future_time = data['input_data']['future_time']

        # Get ground truth
        if target_col and hasattr(future_time, target_col):
            y_true = np.asarray(future_time[target_col], dtype=float)
        else:
            y_true = np.asarray(future_time, dtype=float)
            if y_true.ndim == 2:
                y_true = y_true[:, target_index]

        # Get samples in correct shape (S, T)
        if samples.ndim == 3:
            samples_2d = samples[:, :, target_index]
        else:
            samples_2d = samples

        # Align lengths
        min_len = min(samples_2d.shape[1], len(y_true))
        samples_2d = samples_2d[:, :min_len]
        y_true = y_true[:min_len]

        # Compute CRPS per timestep
        crps_per_timestep = compute_crps(y_true, samples_2d)  # (T,)

        # Define weight scheme
        if weight_scheme == 'clinical_horizons':
            # Clinical-temporal aware: 0-15min:1.0, 15-30min:1.5, 30-60min:2.0, 60-120min:3.0
            if min_len == 24:
                weights = np.array([1.0]*3 + [1.5]*3 + [2.0]*6 + [3.0]*12)
            else:
                # Fallback for different horizon lengths: proportional scaling
                weights = np.linspace(1.0, 3.0, min_len)

        elif weight_scheme == 'hour_based':
            # First hour weight=1.0, second hour weight=2.0
            half = min_len // 2
            weights = np.array([1.0]*half + [2.0]*(min_len - half))

        elif weight_scheme == 'linear':
            # Linear increase from 1 to timestep number
            weights = np.arange(1, min_len + 1, dtype=float)

        else:
            raise ValueError(f"Unknown weight_scheme: {weight_scheme}")

        # Compute weighted average (np.average auto-normalizes: sum(w*x)/sum(w))
        t_crps = float(np.average(crps_per_timestep, weights=weights))

        # Also compute standard CRPS for comparison
        standard_crps = float(np.mean(crps_per_timestep))

        return {
            't_crps': t_crps,
            'crps_standard': standard_crps,
            'n_timesteps': int(min_len)
        }

    except Exception as e:
        print(f"Warning: Failed to compute time-weighted CRPS for {pkl_path}: {e}")
        return None


def compute_clarke_weighted_crps_from_pickle(
    pkl_path: Path,
    target_col: str = None,
    target_index: int = -1
):
    """
    Compute Clarke-weighted CRPS (c_crps) from complete_data.pkl samples.

    This is distribution-aware: evaluates ALL samples (not just median) and assigns
    Clarke zone weights to each sample, then averages. This rewards models whose
    distribution captures dangerous zones when appropriate.

    Clarke weights:
        Zone A (clinically accurate): 1.0
        Zone B (benign errors): 1.5
        Zone C (overcorrection): 3.0
        Zone D (failure to detect): 4.0
        Zone E (erroneous treatment): 5.0

    Args:
        pkl_path: Path to complete_data.pkl
        target_col: Optional column name for target
        target_index: Index to extract from multidimensional targets

    Returns dict with:
        - c_crps: Clarke-weighted CRPS (mg/dL)
        - crps_standard: Standard mean CRPS for comparison (mg/dL)
        - n_timesteps: Number of timesteps
    """
    # Clarke zone weights
    CLARKE_WEIGHTS = {
        'A': 1.0,
        'B': 1.5,
        'C': 3.0,
        'D': 4.0,
        'E': 5.0
    }

    try:
        data = load_complete_data(pkl_path)

        # Extract predictions and ground truth
        samples = np.asarray(data['predictions']['samples'])
        future_time = data['input_data']['future_time']

        # Get ground truth
        if target_col and hasattr(future_time, target_col):
            y_true = np.asarray(future_time[target_col], dtype=float)
        else:
            y_true = np.asarray(future_time, dtype=float)
            if y_true.ndim == 2:
                y_true = y_true[:, target_index]

        # Get samples in correct shape (S, T)
        if samples.ndim == 3:
            samples_2d = samples[:, :, target_index]
        else:
            samples_2d = samples

        # Align lengths
        min_len = min(samples_2d.shape[1], len(y_true))
        samples_2d = samples_2d[:, :min_len]
        y_true = y_true[:min_len]

        # Compute CRPS per timestep
        crps_per_timestep = compute_crps(y_true, samples_2d)  # (T,)

        # Compute Clarke weights per timestep (distribution-aware)
        weighted_sum = 0.0
        total_weight = 0.0

        for t in range(min_len):
            # Get all samples at this timestep
            samples_t = samples_2d[:, t]  # (S,)
            actual_t = y_true[t]

            # Classify each sample into Clarke zone
            weights_for_timestep = []
            for sample_value in samples_t:
                zone = clarke_error_grid_zone(actual_t, sample_value)
                weight = CLARKE_WEIGHTS[zone]
                weights_for_timestep.append(weight)

            # Average weight across all samples at this timestep
            avg_weight = np.mean(weights_for_timestep)

            # Apply to CRPS
            weighted_sum += avg_weight * crps_per_timestep[t]
            total_weight += avg_weight

        # Final Clarke-weighted CRPS
        c_crps = float(weighted_sum / total_weight) if total_weight > 0 else 0.0

        # Also compute standard CRPS for comparison
        standard_crps = float(np.mean(crps_per_timestep))

        return {
            'c_crps': c_crps,
            'crps_standard': standard_crps,
            'n_timesteps': int(min_len)
        }

    except Exception as e:
        print(f"Warning: Failed to compute Clarke-weighted CRPS for {pkl_path}: {e}")
        return None


def compute_clarke_temporal_weighted_crps_from_pickle(
    pkl_path: Path,
    weight_scheme: str = 'clinical_horizons',
    target_col: str = None,
    target_index: int = -1
):
    """
    Compute Clarke-Temporal-weighted CRPS (ct_crps) from complete_data.pkl samples.

    Combines both temporal weighting (later timesteps = more important) and
    distribution-aware Clarke weighting (dangerous zones = more important).

    This is the most comprehensive metric, emphasizing both long-term accuracy
    and clinical safety.

    Args:
        pkl_path: Path to complete_data.pkl
        weight_scheme: Temporal weighting strategy ('clinical_horizons', 'hour_based', 'linear')
        target_col: Optional column name for target
        target_index: Index to extract from multidimensional targets

    Returns dict with:
        - ct_crps: Clarke-Temporal-weighted CRPS (mg/dL)
        - crps_standard: Standard mean CRPS for comparison (mg/dL)
        - n_timesteps: Number of timesteps
    """
    # Clarke zone weights
    CLARKE_WEIGHTS = {
        'A': 1.0,
        'B': 1.5,
        'C': 3.0,
        'D': 4.0,
        'E': 5.0
    }

    try:
        data = load_complete_data(pkl_path)

        # Extract predictions and ground truth
        samples = np.asarray(data['predictions']['samples'])
        future_time = data['input_data']['future_time']

        # Get ground truth
        if target_col and hasattr(future_time, target_col):
            y_true = np.asarray(future_time[target_col], dtype=float)
        else:
            y_true = np.asarray(future_time, dtype=float)
            if y_true.ndim == 2:
                y_true = y_true[:, target_index]

        # Get samples in correct shape (S, T)
        if samples.ndim == 3:
            samples_2d = samples[:, :, target_index]
        else:
            samples_2d = samples

        # Align lengths
        min_len = min(samples_2d.shape[1], len(y_true))
        samples_2d = samples_2d[:, :min_len]
        y_true = y_true[:min_len]

        # Compute CRPS per timestep
        crps_per_timestep = compute_crps(y_true, samples_2d)  # (T,)

        # Define temporal weights
        if weight_scheme == 'clinical_horizons':
            if min_len == 24:
                time_weights = np.array([1.0]*3 + [1.5]*3 + [2.0]*6 + [3.0]*12)
            else:
                time_weights = np.linspace(1.0, 3.0, min_len)
        elif weight_scheme == 'hour_based':
            half = min_len // 2
            time_weights = np.array([1.0]*half + [2.0]*(min_len - half))
        elif weight_scheme == 'linear':
            time_weights = np.arange(1, min_len + 1, dtype=float)
        else:
            raise ValueError(f"Unknown weight_scheme: {weight_scheme}")

        # Compute combined Clarke-Temporal weights per timestep
        weighted_sum = 0.0
        total_weight = 0.0

        for t in range(min_len):
            # Get all samples at this timestep
            samples_t = samples_2d[:, t]  # (S,)
            actual_t = y_true[t]

            # Classify each sample into Clarke zone
            clarke_weights_for_timestep = []
            for sample_value in samples_t:
                zone = clarke_error_grid_zone(actual_t, sample_value)
                weight = CLARKE_WEIGHTS[zone]
                clarke_weights_for_timestep.append(weight)

            # Average Clarke weight across all samples at this timestep
            avg_clarke_weight = np.mean(clarke_weights_for_timestep)

            # Combine temporal weight × Clarke weight
            combined_weight = time_weights[t] * avg_clarke_weight

            # Apply to CRPS
            weighted_sum += combined_weight * crps_per_timestep[t]
            total_weight += combined_weight

        # Final Clarke-Temporal-weighted CRPS
        ct_crps = float(weighted_sum / total_weight) if total_weight > 0 else 0.0

        # Also compute standard CRPS for comparison
        standard_crps = float(np.mean(crps_per_timestep))

        return {
            'ct_crps': ct_crps,
            'crps_standard': standard_crps,
            'n_timesteps': int(min_len)
        }

    except Exception as e:
        print(f"Warning: Failed to compute Clarke-Temporal-weighted CRPS for {pkl_path}: {e}")
        return None


def collect_clinical_metrics(result_dir: Path, low_mgdl: float = 70.0, high_mgdl: float = 180.0):
    """
    Scan all complete_data.pkl files and compute clinical metrics.

    Returns DataFrame with columns:
        - model
        - task
        - seed
        - crps (standard CRPS, units: mg/dL)
        - t_crps (temporal-weighted CRPS, units: mg/dL)
        - c_crps (Clarke-weighted CRPS, units: mg/dL)
        - ct_crps (Clarke-Temporal-weighted CRPS, units: mg/dL)
        - clarke_a, clarke_b, clarke_c, clarke_d, clarke_e (%)
        - clarke_ab (% clinically acceptable, = clarke_a + clarke_b)
        - clarke_cde (% clinically problematic, = clarke_c + clarke_d + clarke_e)
        - mae (mg/dL)
        - rmse (mg/dL)
    """
    records = []

    for model_dir in sorted(result_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith('_'):
            continue

        model_name = model_dir.name
        print(f"\nProcessing model: {model_name}")

        for task_dir in sorted(model_dir.iterdir()):
            if not task_dir.is_dir():
                continue

            task_name = task_dir.name

            for seed_dir in sorted(task_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue

                seed_id = seed_dir.name

                # Load complete_data.pkl
                pkl_file = seed_dir / 'complete_data.pkl'
                if not pkl_file.exists():
                    continue

                # Compute Clarke Error Grid
                clarke = compute_clarke_from_pickle(pkl_file)

                # Compute MAE/RMSE
                mae_rmse = compute_mae_rmse_from_pickle(pkl_file)

                # Compute CRPS from samples (using util_crps.py)
                crps_result = compute_crps_from_pickle(pkl_file)
                crps = crps_result['crps'] if crps_result else None

                # Compute temporal-weighted CRPS (t_crps)
                t_crps_result = compute_time_weighted_crps_from_pickle(
                    pkl_file, weight_scheme='clinical_horizons'
                )
                t_crps = t_crps_result['t_crps'] if t_crps_result else None

                # Compute Clarke-weighted CRPS (c_crps)
                c_crps_result = compute_clarke_weighted_crps_from_pickle(pkl_file)
                c_crps = c_crps_result['c_crps'] if c_crps_result else None

                # Compute Clarke-Temporal-weighted CRPS (ct_crps)
                ct_crps_result = compute_clarke_temporal_weighted_crps_from_pickle(
                    pkl_file, weight_scheme='clinical_horizons'
                )
                ct_crps = ct_crps_result['ct_crps'] if ct_crps_result else None

                # Build record
                record = {
                    'model': model_name,
                    'task': task_name,
                    'seed': seed_id,
                    'crps': crps,
                    't_crps': t_crps,
                    'c_crps': c_crps,
                    'ct_crps': ct_crps
                }

                if clarke:
                    record['clarke_a'] = clarke['clarke_a']
                    record['clarke_b'] = clarke['clarke_b']
                    record['clarke_c'] = clarke['clarke_c']
                    record['clarke_d'] = clarke['clarke_d']
                    record['clarke_e'] = clarke['clarke_e']
                    record['clarke_ab'] = clarke['clarke_ab']
                    record['clarke_cde'] = clarke['clarke_cde']

                if mae_rmse:
                    record['mae'] = mae_rmse['mae']
                    record['rmse'] = mae_rmse['rmse']

                records.append(record)

        print(f"  Collected {len([r for r in records if r['model'] == model_name])} instances")

    return pd.DataFrame(records)


def aggregate_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate clinical metrics by model (across all tasks and seeds).

    Returns summary with mean ± std for each metric.
    """
    summary = []

    for model in sorted(df['model'].unique()):
        model_data = df[df['model'] == model]

        record = {'model': model, 'n_instances': len(model_data)}

        for metric in ['crps', 't_crps', 'c_crps', 'ct_crps', 'clarke_a', 'clarke_b', 'clarke_c', 'clarke_d', 'clarke_e', 'clarke_ab', 'clarke_cde', 'mae', 'rmse']:
            if metric in model_data.columns:
                values = model_data[metric].dropna()
                if len(values) > 0:
                    record[f'{metric}_mean'] = values.mean()
                    record[f'{metric}_std'] = values.std()
                    record[f'{metric}_count'] = len(values)

        summary.append(record)

    return pd.DataFrame(summary)


def main():
    """Run clinical metrics computation."""
    print("=" * 80)
    print("CLINICAL METRICS COMPUTATION")
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
    print("   This may take several minutes...")

    # Collect metrics
    df = collect_clinical_metrics(result_dir, low_mgdl=70.0, high_mgdl=180.0)

    if df.empty:
        print("Error: No results found!")
        return

    print(f"\n2. Collected {len(df)} instances across {df['model'].nunique()} models")

    # Save detailed results
    detailed_file = output_dir / 'clinical_metrics_detailed.csv'
    df.to_csv(detailed_file, index=False)
    print(f"   Saved detailed results: {detailed_file}")

    # Aggregate by model
    print("\n3. Aggregating by model...")
    summary_df = aggregate_by_model(df)

    # Sort by CRPS (primary metric)
    if 'crps_mean' in summary_df.columns:
        summary_df = summary_df.sort_values('crps_mean')

    summary_file = output_dir / 'clinical_metrics_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"   Saved summary: {summary_file}")

    # Create report
    report_file = output_dir / 'clinical_metrics_report.txt'
    with open(report_file, 'w') as f:
        f.write("CLINICAL METRICS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("TOP 10 MODELS BY CRPS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<45} {'CRPS':>8} {'Clarke':>8} {'MAE':>8} {'RMSE':>8}\n")
        f.write("-" * 80 + "\n")

        top10 = summary_df.head(10)
        for _, row in top10.iterrows():
            crps = row.get('crps_mean', float('nan'))
            clarke_ab = row.get('clarke_ab_mean', float('nan'))
            mae = row.get('mae_mean', float('nan'))
            rmse = row.get('rmse_mean', float('nan'))

            f.write(f"{row['model']:<45} {crps:8.4f} {clarke_ab:7.2f}% {mae:8.2f} {rmse:8.2f}\n")

        f.write("\n\nMETRIC CORRELATIONS\n")
        f.write("-" * 80 + "\n")

        # Compute correlations
        metric_cols = [col for col in summary_df.columns if col.endswith('_mean') and 'clarke' not in col]
        metric_cols = [col for col in metric_cols if col in ['crps_mean', 'mae_mean', 'rmse_mean']]
        if 'clarke_ab_mean' in summary_df.columns:
            metric_cols.append('clarke_ab_mean')

        if len(metric_cols) >= 2:
            corr_data = summary_df[metric_cols].corr()
            f.write("\nPearson Correlation Matrix:\n")
            f.write(corr_data.to_string())
            f.write("\n")

    print(f"   Saved report: {report_file}")

    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    if 'crps_mean' in summary_df.columns and 'clarke_ab_mean' in summary_df.columns:
        crps_clarke_corr = summary_df['crps_mean'].corr(summary_df['clarke_ab_mean'])
        print(f"\nCRPS vs Clarke A+B correlation: {crps_clarke_corr:.3f}")

    if 'crps_mean' in summary_df.columns and 'mae_mean' in summary_df.columns:
        crps_mae_corr = summary_df['crps_mean'].corr(summary_df['mae_mean'])
        print(f"CRPS vs MAE correlation: {crps_mae_corr:.3f}")

    print("\nTop 3 models by CRPS:")
    for i, (_, row) in enumerate(summary_df.head(3).iterrows(), 1):
        print(f"  {i}. {row['model']}")
        if 'crps_mean' in row:
            print(f"      CRPS: {row['crps_mean']:.4f}")
        if 'clarke_ab_mean' in row:
            print(f"      Clarke A+B: {row['clarke_ab_mean']:.2f}%")
        if 'mae_mean' in row:
            print(f"      MAE: {row['mae_mean']:.2f} mg/dL")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - {detailed_file}")
    print(f"  - {summary_file}")
    print(f"  - {report_file}")


if __name__ == '__main__':
    main()
