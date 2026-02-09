"""
Ablation Study for Glucose-RCRPS Parameter Tuning
==================================================

Tests different combinations of:
- β (constraint penalty weight): controls penalty term magnitude
- Clarke zone weights: controls clinical RoI emphasis

Goal: Find parameters that maximize correlation with Clarke A+B (target < -0.5)
while maintaining balanced component contributions (Penalty:Weighted ratio ~1:1 to 2:1)

Usage:
    python evaluation/eval_fn/ablation_glucose_rcrps.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from evaluation.measurements import load_complete_data
from evaluation.eval_fn.compute_glucose_rcrps import (
    compute_clarke_weighted_crps,
    compute_constraint_crps,
    DEFAULT_ALPHA
)
from eventglucose.metrics.crps import crps


# Configuration sets to test
ABLATION_CONFIGS = {
    'current': {
        'beta': 10.0,
        'clarke_weights': {'A': 1.0, 'B': 1.5, 'C': 3.0, 'D': 4.0, 'E': 5.0},
        'description': 'Baseline (high β, mild weights)'
    },
    'reduce_beta': {
        'beta': 2.0,
        'clarke_weights': {'A': 1.0, 'B': 1.5, 'C': 3.0, 'D': 4.0, 'E': 5.0},
        'description': 'Reduce β only (β=2)'
    },
    'increase_weights': {
        'beta': 10.0,
        'clarke_weights': {'A': 1.0, 'B': 2.0, 'C': 8.0, 'D': 12.0, 'E': 15.0},
        'description': 'Increase weights only (aggressive zones)'
    },
    'balanced': {
        'beta': 2.0,
        'clarke_weights': {'A': 1.0, 'B': 2.0, 'C': 8.0, 'D': 12.0, 'E': 15.0},
        'description': '**RECOMMENDED** (β=2, aggressive weights)'
    },
    'low_beta': {
        'beta': 1.0,
        'clarke_weights': {'A': 1.0, 'B': 2.0, 'C': 8.0, 'D': 12.0, 'E': 15.0},
        'description': 'Very low β (β=1, aggressive weights)'
    },
    'very_aggressive': {
        'beta': 2.0,
        'clarke_weights': {'A': 1.0, 'B': 3.0, 'C': 10.0, 'D': 15.0, 'E': 20.0},
        'description': 'Very aggressive weights (β=2, max emphasis on danger)'
    },
}


def compute_glucose_rcrps_custom(
    crps_values: np.ndarray,
    actual_values: np.ndarray,
    predicted_values: np.ndarray,
    predicted_samples: np.ndarray,
    clarke_weights: Dict[str, float],
    alpha: float = DEFAULT_ALPHA,
    beta: float = 10.0
) -> Dict:
    """
    Compute Glucose-RCRPS with custom parameters.

    Same as compute_glucose_rcrps() but allows custom Clarke weights and β.
    """
    # Import here to avoid circular dependency
    from evaluation.eval_fn.compute_glucose_rcrps import CLARKE_WEIGHTS

    # Temporarily override CLARKE_WEIGHTS
    original_weights = CLARKE_WEIGHTS.copy()
    CLARKE_WEIGHTS.update(clarke_weights)

    try:
        # Component 1: Clarke-weighted CRPS with custom weights
        clarke_crps, zone_breakdown, zone_counts = compute_clarke_weighted_crps(
            crps_values, actual_values, predicted_values
        )

        # Component 2: Constraint CRPS (unchanged)
        constraint_crps_value = compute_constraint_crps(
            predicted_samples, actual_values
        )

        # Component 3: Assembly with custom β
        weighted_term = alpha * clarke_crps
        penalty_term = alpha * beta * constraint_crps_value
        glucose_rcrps = weighted_term + penalty_term

        return {
            'glucose_rcrps': glucose_rcrps,
            'clarke_weighted_crps': clarke_crps,
            'constraint_crps': constraint_crps_value,
            'weighted_term': weighted_term,
            'penalty_term': penalty_term,
            'alpha': alpha,
            'beta': beta,
            'zone_breakdown': zone_breakdown,
            'zone_counts': zone_counts
        }
    finally:
        # Restore original weights
        CLARKE_WEIGHTS.update(original_weights)


def extract_prediction_data(pkl_path: Path, target_index: int = -1):
    """Extract prediction data from complete_data.pkl."""
    try:
        data = load_complete_data(pkl_path)

        samples = np.asarray(data['predictions']['samples'])
        future_time = data['input_data']['future_time']

        # Get ground truth
        y_true = np.asarray(future_time, dtype=float)
        if y_true.ndim == 2:
            y_true = y_true[:, target_index]

        # Extract samples for target variable
        if samples.ndim == 3:
            samples_2d = samples[:, :, target_index]
        else:
            samples_2d = samples

        # Align lengths
        min_len = min(samples_2d.shape[1], len(y_true))
        samples_2d = samples_2d[:, :min_len]
        y_true = y_true[:min_len]

        # Compute point forecast and CRPS
        y_pred = np.median(samples_2d, axis=0)
        crps_values = crps(y_true, samples_2d)

        return {
            'actual_values': y_true,
            'predicted_values': y_pred,
            'predicted_samples': samples_2d,
            'crps_values': crps_values
        }
    except Exception as e:
        return None


def run_ablation_study(result_dir: Path, csv_path: Path):
    """
    Run ablation study on all configurations.

    Returns DataFrame with results for each configuration.
    """
    print("=" * 80)
    print("GLUCOSE-RCRPS ABLATION STUDY")
    print("=" * 80)

    # Load existing CSV with metadata
    print(f"\n1. Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Found {len(df)} rows")

    # Store results for each configuration
    config_results = {}

    for config_name, config in ABLATION_CONFIGS.items():
        print(f"\n2. Testing configuration: {config_name}")
        print(f"   {config['description']}")
        print(f"   β = {config['beta']}, weights = {config['clarke_weights']}")

        glucose_rcrps_values = []
        weighted_term_values = []
        penalty_term_values = []

        # Compute for each row
        for idx, row in df.iterrows():
            model = row['model']
            task = row['task']
            seed = str(row['seed'])

            # Build path to complete_data.pkl
            pkl_path = result_dir / model / task / seed / 'complete_data.pkl'

            if not pkl_path.exists():
                glucose_rcrps_values.append(np.nan)
                weighted_term_values.append(np.nan)
                penalty_term_values.append(np.nan)
                continue

            # Extract data
            pred_data = extract_prediction_data(pkl_path)
            if pred_data is None:
                glucose_rcrps_values.append(np.nan)
                weighted_term_values.append(np.nan)
                penalty_term_values.append(np.nan)
                continue

            # Compute with custom parameters
            try:
                results = compute_glucose_rcrps_custom(
                    crps_values=pred_data['crps_values'],
                    actual_values=pred_data['actual_values'],
                    predicted_values=pred_data['predicted_values'],
                    predicted_samples=pred_data['predicted_samples'],
                    clarke_weights=config['clarke_weights'],
                    beta=config['beta']
                )
                glucose_rcrps_values.append(results['glucose_rcrps'])
                weighted_term_values.append(results['weighted_term'])
                penalty_term_values.append(results['penalty_term'])
            except Exception as e:
                glucose_rcrps_values.append(np.nan)
                weighted_term_values.append(np.nan)
                penalty_term_values.append(np.nan)

        # Store results
        config_results[config_name] = {
            'glucose_rcrps': glucose_rcrps_values,
            'weighted_term': weighted_term_values,
            'penalty_term': penalty_term_values,
            'beta': config['beta'],
            'weights': config['clarke_weights'],
            'description': config['description']
        }

        print(f"   Computed {sum(1 for v in glucose_rcrps_values if not np.isnan(v))} / {len(df)} values")

    return df, config_results


def analyze_configurations(df: pd.DataFrame, config_results: Dict):
    """
    Analyze all configurations and rank by correlation with Clarke A+B.
    """
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)

    results_summary = []

    for config_name, results in config_results.items():
        # Create temporary dataframe with this configuration
        temp_df = df.copy()
        temp_df['glucose_rcrps_test'] = results['glucose_rcrps']
        temp_df['weighted_term_test'] = results['weighted_term']
        temp_df['penalty_term_test'] = results['penalty_term']

        # Filter valid values
        valid = temp_df[temp_df['glucose_rcrps_test'].notna()]

        # Compute statistics
        mean_grcrps = valid['glucose_rcrps_test'].mean()
        std_grcrps = valid['glucose_rcrps_test'].std()
        mean_weighted = valid['weighted_term_test'].mean()
        mean_penalty = valid['penalty_term_test'].mean()

        # Compute ratio (avoid division by zero)
        if mean_weighted > 0:
            penalty_weighted_ratio = mean_penalty / mean_weighted
        else:
            penalty_weighted_ratio = np.inf

        # Compute correlations
        corr_with_crps = valid[['crps', 'glucose_rcrps_test']].corr().iloc[0, 1] if 'crps' in valid.columns else np.nan
        corr_with_clarke = valid[['clarke_ab', 'glucose_rcrps_test']].corr().iloc[0, 1] if 'clarke_ab' in valid.columns else np.nan

        results_summary.append({
            'config': config_name,
            'description': results['description'],
            'beta': results['beta'],
            'weights_C': results['weights']['C'],
            'weights_D': results['weights']['D'],
            'weights_E': results['weights']['E'],
            'mean_grcrps': mean_grcrps,
            'std_grcrps': std_grcrps,
            'mean_weighted': mean_weighted,
            'mean_penalty': mean_penalty,
            'penalty_weighted_ratio': penalty_weighted_ratio,
            'corr_crps': corr_with_crps,
            'corr_clarke_ab': corr_with_clarke,
            'n_valid': len(valid)
        })

    # Convert to DataFrame and sort by Clarke correlation (more negative is better)
    summary_df = pd.DataFrame(results_summary)
    summary_df = summary_df.sort_values('corr_clarke_ab')

    # Print summary table
    print("\nConfiguration Comparison (sorted by Clarke A+B correlation):")
    print("-" * 80)
    print(f"{'Config':<20} {'β':<6} {'C/D/E':<12} {'Clarke Corr':<12} {'Ratio':<10} {'Description'}")
    print("-" * 80)

    for _, row in summary_df.iterrows():
        weights_str = f"{int(row['weights_C'])}/{int(row['weights_D'])}/{int(row['weights_E'])}"
        print(f"{row['config']:<20} {row['beta']:<6.1f} {weights_str:<12} {row['corr_clarke_ab']:<12.4f} {row['penalty_weighted_ratio']:<10.2f} {row['description']}")

    # Detailed statistics for best configuration
    best_config = summary_df.iloc[0]
    print(f"\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    print(f"\nConfig: {best_config['config']}")
    print(f"Description: {best_config['description']}")
    print(f"\nParameters:")
    print(f"  β = {best_config['beta']}")
    print(f"  Clarke weights: C={int(best_config['weights_C'])}, D={int(best_config['weights_D'])}, E={int(best_config['weights_E'])}")
    print(f"\nPerformance:")
    print(f"  Clarke A+B correlation: {best_config['corr_clarke_ab']:.4f} (target: < -0.5)")
    print(f"  CRPS correlation: {best_config['corr_crps']:.4f}")
    print(f"  Penalty:Weighted ratio: {best_config['penalty_weighted_ratio']:.2f} (target: 1-2)")
    print(f"\nComponent Magnitudes:")
    print(f"  Mean weighted term: {best_config['mean_weighted']:.4f}")
    print(f"  Mean penalty term: {best_config['mean_penalty']:.4f}")
    print(f"  Mean Glucose-RCRPS: {best_config['mean_grcrps']:.4f} ± {best_config['std_grcrps']:.4f}")

    # Comparison with baseline
    baseline = summary_df[summary_df['config'] == 'current'].iloc[0]
    print(f"\nImprovement vs Baseline:")
    print(f"  Clarke correlation: {baseline['corr_clarke_ab']:.4f} → {best_config['corr_clarke_ab']:.4f}")
    improvement = abs(best_config['corr_clarke_ab']) - abs(baseline['corr_clarke_ab'])
    print(f"  Improvement: {improvement:+.4f} ({100*improvement/abs(baseline['corr_clarke_ab']):+.1f}%)")
    print(f"  Ratio improvement: {baseline['penalty_weighted_ratio']:.2f} → {best_config['penalty_weighted_ratio']:.2f}")

    return summary_df, best_config


def update_csv_with_best_config(df: pd.DataFrame, config_results: Dict, best_config_name: str, output_path: Path):
    """
    Update CSV with the best configuration's Glucose-RCRPS values.
    """
    print(f"\n" + "=" * 80)
    print("UPDATING CSV WITH BEST CONFIGURATION")
    print("=" * 80)

    best_results = config_results[best_config_name]

    # Update columns
    df['glucose_rcrps'] = best_results['glucose_rcrps']
    df['weighted_term'] = best_results['weighted_term']
    df['penalty_term'] = best_results['penalty_term']

    # Add metadata columns
    df['glucose_rcrps_beta'] = best_results['beta']
    df['glucose_rcrps_config'] = best_config_name

    # Save
    df.to_csv(output_path, index=False)
    print(f"\nUpdated CSV saved: {output_path}")
    print(f"  Configuration: {best_config_name}")
    print(f"  β = {best_results['beta']}")
    print(f"  Clarke weights: {best_results['weights']}")


def main():
    """Run ablation study and update CSV with best configuration."""
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

    # Run ablation study
    df, config_results = run_ablation_study(result_dir, csv_path)

    # Analyze and rank configurations
    summary_df, best_config = analyze_configurations(df, config_results)

    # Save summary
    summary_path = repo_root / 'evaluation' / 'results' / 'glucose_rcrps_ablation.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nAblation summary saved: {summary_path}")

    # Update CSV with best configuration
    best_config_name = best_config['config']
    update_csv_with_best_config(df, config_results, best_config_name, csv_path)

    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
