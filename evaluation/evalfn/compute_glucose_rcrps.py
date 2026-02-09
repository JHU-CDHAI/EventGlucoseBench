"""
Glucose-RCRPS Computation
==========================

Compute Glucose-RCRPS (Region-of-Interest CRPS adapted for clinical glucose forecasting)
by combining three components from the CiK benchmark's RCRPS framework:

1. **Component 1 - RoI Weighting**: Clarke Error Grid zones define clinical RoI
   - Dangerous zones (C/D/E) get 3-5× higher weight than safe zones (A/B)

2. **Component 2 - Constraint Penalties**: Physiological violation penalties
   - Rate mismatch: |Δpred - Δactual| > 20 mg/dL per 5-min interval
   - Absolute bounds: glucose < 40 or > 400 mg/dL

3. **Component 3 - Scale Normalization**: Cross-patient comparability
   - α = 1/110 (for 70-180 mg/dL typical range)

Formula:
    Glucose-RCRPS = α · [Clarke-weighted CRPS + β · CRPS(violations)]

where β = 10 (following CiK benchmark)

Usage:
    python 0--evaluation/evalfn/compute_glucose_rcrps.py

References:
    - CiK benchmark (RCRPS): Context is Key paper
    - Clarke Error Grid: Clarke et al. (1987) Diabetes Care
    - CRPS: Gneiting & Raftery (2007)
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Any

# Add parent directory to path for imports
repo_root = Path(__file__).resolve().parent.parent.parent

import numpy as np
import pandas as pd
from evalfn.clinical_metrics import clarke_error_grid_zone
from evalfn.util_crps import crps

# Clarke zone weights (clinical priority)
CLARKE_WEIGHTS = {
    'A': 1.0,  # Clinically accurate - baseline
    'B': 1.5,  # Benign error - slightly elevated
    'C': 3.0,  # Overcorrection - dangerous
    'D': 4.0,  # Failure to detect - very dangerous
    'E': 5.0,  # Erroneous treatment - extremely dangerous
}

# Glucose-RCRPS parameters
DEFAULT_ALPHA = 1.0 / 110.0  # Scale normalization: 1/(180-70) typical glucose range
DEFAULT_BETA = 10.0           # Constraint penalty weight (following CiK)
RATE_THRESHOLD = 20.0         # mg/dL per 5-min interval
LOWER_BOUND = 40.0            # mg/dL (physiologically impossible below)
UPPER_BOUND = 400.0           # mg/dL (physiologically impossible above)


def compute_clarke_weighted_crps(
    crps_values: np.ndarray,      # shape: (n_timesteps,)
    actual_values: np.ndarray,    # shape: (n_timesteps,)
    predicted_values: np.ndarray  # shape: (n_timesteps,)
) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    """
    Compute Clarke zone-weighted CRPS (Component 1 of Glucose-RCRPS).

    This implements the RoI weighting component by assigning higher weights
    to predictions in clinically dangerous Clarke zones.

    Args:
        crps_values: CRPS at each timestep (n_timesteps,)
        actual_values: Ground truth glucose trajectory (n_timesteps,)
        predicted_values: Point forecast trajectory (n_timesteps,)

    Returns:
        weighted_crps: Clarke-weighted CRPS score (float)
        zone_breakdown: Mean CRPS per zone, e.g. {'A': 2.5, 'B': 3.1, ...}
        zone_counts: Number of timesteps per zone, e.g. {'A': 15, 'B': 8, ...}

    Implementation:
        1. For each timestep i:
           a. Assign zone = assign_clarke_zone(actual[i], predicted[i])
           b. Apply weight = CLARKE_WEIGHTS[zone]
           c. Accumulate weighted_sum += weight * crps_values[i]
           d. Accumulate total_weight += weight
        2. weighted_crps = weighted_sum / total_weight
        3. Compute mean CRPS per zone for breakdown analysis
    """
    n_timesteps = len(crps_values)
    assert len(actual_values) == n_timesteps
    assert len(predicted_values) == n_timesteps

    # Initialize accumulators
    weighted_sum = 0.0
    total_weight = 0.0
    zone_crps = {z: [] for z in CLARKE_WEIGHTS.keys()}

    # Compute Clarke zone for each timestep and apply weights
    for i in range(n_timesteps):
        # Determine Clarke zone for this prediction
        zone = clarke_error_grid_zone(actual_values[i], predicted_values[i])

        # Apply clinical weight
        weight = CLARKE_WEIGHTS[zone]
        weighted_sum += weight * crps_values[i]
        total_weight += weight

        # Track CRPS by zone for breakdown analysis
        zone_crps[zone].append(crps_values[i])

    # Compute weighted average
    if total_weight > 0:
        weighted_crps = weighted_sum / total_weight
    else:
        weighted_crps = 0.0

    # Compute mean CRPS per zone
    zone_breakdown = {
        zone: np.mean(values) if len(values) > 0 else 0.0
        for zone, values in zone_crps.items()
    }

    # Count timesteps per zone
    zone_counts = {
        zone: len(values)
        for zone, values in zone_crps.items()
    }

    return weighted_crps, zone_breakdown, zone_counts


def compute_rate_mismatch_violations(
    predicted_samples: np.ndarray,  # shape: (n_samples, n_timesteps)
    actual_values: np.ndarray       # shape: (n_timesteps,)
) -> np.ndarray:
    """
    Compute rate mismatch violations between predicted and actual trajectories.

    Penalizes predicting rapid changes when stable, or stable when rapid changes occur.
    This captures models with poor understanding of glucose dynamics.

    Args:
        predicted_samples: Full forecast distribution (n_samples × n_timesteps)
        actual_values: Ground truth glucose trajectory (n_timesteps,)

    Returns:
        violations: Array of shape (n_samples,), mean violation amount per sample

    Implementation:
        1. Compute rate of change (5-min intervals):
           Δpred = predicted_samples[:, 1:] - predicted_samples[:, :-1]
               # shape: (n_samples, n_timesteps-1)
           Δactual = actual_values[1:] - actual_values[:-1]
               # shape: (n_timesteps-1,)

        2. Rate mismatch for each sample at each timestep:
           mismatch = |Δpred - Δactual|  # broadcast Δactual across samples

        3. Apply threshold (20 mg/dL per 5-min):
           violations = max(0, mismatch - 20)

        4. Average across timesteps per sample:
           return mean(violations, axis=1)  # shape: (n_samples,)

    Example:
        - Predict: +25 mg/dL, Actual: +3 mg/dL → mismatch = 22 → violation = 2
        - Predict: +18 mg/dL, Actual: +20 mg/dL → mismatch = 2 → violation = 0
    """
    n_samples, n_timesteps = predicted_samples.shape
    assert len(actual_values) == n_timesteps

    if n_timesteps < 2:
        # Cannot compute rate with single timestep
        return np.zeros(n_samples)

    # Compute rate of change (difference between consecutive timesteps)
    # Shape: (n_samples, n_timesteps-1)
    delta_pred = predicted_samples[:, 1:] - predicted_samples[:, :-1]

    # Shape: (n_timesteps-1,)
    delta_actual = actual_values[1:] - actual_values[:-1]

    # Compute rate mismatch: |Δpred - Δactual|
    # Broadcasting delta_actual across samples
    # Shape: (n_samples, n_timesteps-1)
    rate_mismatch = np.abs(delta_pred - delta_actual[np.newaxis, :])

    # Apply threshold: violations = max(0, mismatch - threshold)
    # Penalize only mismatches exceeding 20 mg/dL per 5-min
    violations = np.maximum(0, rate_mismatch - RATE_THRESHOLD)

    # Average violation across timesteps per sample
    # Shape: (n_samples,)
    mean_violations = np.mean(violations, axis=1)

    return mean_violations


def compute_bound_violations(
    predicted_samples: np.ndarray  # shape: (n_samples, n_timesteps)
) -> np.ndarray:
    """
    Compute violations of physiological glucose bounds.

    Penalizes predictions that are physiologically impossible.

    Args:
        predicted_samples: Full forecast distribution (n_samples × n_timesteps)

    Returns:
        violations: Array of shape (n_samples,), mean violation per sample

    Implementation:
        1. Lower bound violations (glucose < 40 mg/dL):
           lower_viol = max(0, 40 - predicted_samples)

        2. Upper bound violations (glucose > 400 mg/dL):
           upper_viol = max(0, predicted_samples - 400)

        3. Total violation per sample:
           total = lower_viol + upper_viol  # shape: (n_samples, n_timesteps)
           return mean(total, axis=1)        # shape: (n_samples,)
    """
    # Lower bound violations: how far below 40 mg/dL
    # Shape: (n_samples, n_timesteps)
    lower_violations = np.maximum(0, LOWER_BOUND - predicted_samples)

    # Upper bound violations: how far above 400 mg/dL
    # Shape: (n_samples, n_timesteps)
    upper_violations = np.maximum(0, predicted_samples - UPPER_BOUND)

    # Total violations
    # Shape: (n_samples, n_timesteps)
    total_violations = lower_violations + upper_violations

    # Average across timesteps per sample
    # Shape: (n_samples,)
    mean_violations = np.mean(total_violations, axis=1)

    return mean_violations


def compute_constraint_crps(
    predicted_samples: np.ndarray,  # shape: (n_samples, n_timesteps)
    actual_values: np.ndarray       # shape: (n_timesteps,)
) -> float:
    """
    Compute CRPS on constraint violations (Component 2 of Glucose-RCRPS).

    Since ground truth has 0 violations (real glucose obeys physics), we compute
    CRPS(violation_samples, 0) to penalize predictions that violate constraints.

    Args:
        predicted_samples: Full forecast distribution (n_samples × n_timesteps)
        actual_values: Ground truth trajectory (n_timesteps,)

    Returns:
        constraint_crps: CRPS penalty for constraint violations (float)

    Implementation:
        1. Compute rate mismatch violations:
           rate_viol = compute_rate_mismatch_violations(predicted_samples, actual_values)
               # shape: (n_samples,)

        2. Compute bound violations:
           bound_viol = compute_bound_violations(predicted_samples)
               # shape: (n_samples,)

        3. Total violations per sample:
           total_viol = rate_viol + bound_viol  # shape: (n_samples,)

        4. Compute CRPS(total_viol, 0):
           Ground truth = 0 (no violations)
           Use existing crps() function with target=0

    Note: Returns 0 if no violations across all samples (perfect constraint satisfaction)
    """
    # Compute rate mismatch violations per sample
    # Shape: (n_samples,)
    rate_violations = compute_rate_mismatch_violations(predicted_samples, actual_values)

    # Compute bound violations per sample
    # Shape: (n_samples,)
    bound_violations = compute_bound_violations(predicted_samples)

    # Total violations per sample
    # Shape: (n_samples,)
    total_violations = rate_violations + bound_violations

    # If no violations at all, return 0
    if np.sum(total_violations) == 0:
        return 0.0

    # Compute CRPS with ground truth = 0 (no violations)
    # The crps() function expects:
    #   - target: scalar or array (we use scalar 0)
    #   - samples: (n_samples, ...) array
    # We need to expand dims to make total_violations 2D: (n_samples, 1)
    total_violations_2d = total_violations[:, np.newaxis]
    target = np.array([0.0])  # Ground truth: no violations

    # Compute CRPS
    # Returns array of shape (1,) since target has shape (1,)
    constraint_crps_value = crps(target, total_violations_2d)[0]

    return float(constraint_crps_value)


def compute_glucose_rcrps(
    crps_values: np.ndarray,       # shape: (n_timesteps,)
    actual_values: np.ndarray,     # shape: (n_timesteps,)
    predicted_values: np.ndarray,  # shape: (n_timesteps,)
    predicted_samples: np.ndarray, # shape: (n_samples, n_timesteps)
    alpha: float = DEFAULT_ALPHA,  # Scale normalization
    beta: float = DEFAULT_BETA     # Constraint penalty weight
) -> Dict[str, Any]:
    """
    Compute full Glucose-RCRPS with all 3 components.

    Adapts CiK's RCRPS framework for clinical glucose forecasting:
    - Component 1: Clarke zone RoI weighting (temporal → value-based RoI)
    - Component 2: Physiological constraint penalties (rate + bounds)
    - Component 3: Scale normalization (cross-patient comparability)

    Args:
        crps_values: Per-timestep CRPS scores (n_timesteps,)
        actual_values: Ground truth glucose trajectory (n_timesteps,)
        predicted_values: Point forecast (median of samples) (n_timesteps,)
        predicted_samples: Full predictive distribution (n_samples, n_timesteps)
        alpha: Normalization factor (default: 1/110 for 70-180 mg/dL range)
        beta: Constraint penalty multiplier (default: 10, following CiK)

    Returns:
        Dictionary containing:
        {
            'glucose_rcrps': float,              # Final score
            'clarke_weighted_crps': float,       # Component 1
            'constraint_crps': float,            # Component 2 (before β)
            'weighted_term': float,              # α * clarke_crps
            'penalty_term': float,               # α * β * constraint_crps
            'alpha': float,                      # Normalization used
            'beta': float,                       # Constraint weight used
            'zone_breakdown': Dict[str, float],  # CRPS per Clarke zone
            'zone_counts': Dict[str, int]        # Timesteps per zone
        }

    Formula:
        Glucose-RCRPS = α · [clarke_weighted_crps + β · constraint_crps]

    Implementation:
        # Component 1: Clarke-weighted CRPS (RoI)
        clarke_crps, zone_breakdown, zone_counts = compute_clarke_weighted_crps(
            crps_values, actual_values, predicted_values
        )

        # Component 2: Constraint CRPS
        constraint_crps_value = compute_constraint_crps(
            predicted_samples, actual_values
        )

        # Component 3: Scale normalization + final assembly
        weighted_term = alpha * clarke_crps
        penalty_term = alpha * beta * constraint_crps_value
        glucose_rcrps = weighted_term + penalty_term
    """
    # Component 1: Clarke-weighted CRPS (RoI)
    clarke_crps, zone_breakdown, zone_counts = compute_clarke_weighted_crps(
        crps_values, actual_values, predicted_values
    )

    # Component 2: Constraint CRPS
    constraint_crps_value = compute_constraint_crps(
        predicted_samples, actual_values
    )

    # Component 3: Scale normalization + final assembly
    weighted_term = alpha * clarke_crps
    penalty_term = alpha * beta * constraint_crps_value
    glucose_rcrps = weighted_term + penalty_term

    # Package results with component breakdown
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


def validate_results(results: Dict[str, Any]) -> None:
    """
    Verify Glucose-RCRPS computation correctness.

    Checks:
    1. glucose_rcrps >= 0 (CRPS is non-negative)
    2. constraint_crps >= 0 (violations are non-negative)
    3. All zone breakdowns >= 0

    Raises AssertionError if any check fails
    """
    assert results['glucose_rcrps'] >= 0, f"glucose_rcrps must be >= 0, got {results['glucose_rcrps']}"
    assert results['constraint_crps'] >= 0, f"constraint_crps must be >= 0, got {results['constraint_crps']}"
    assert results['clarke_weighted_crps'] >= 0, f"clarke_weighted_crps must be >= 0, got {results['clarke_weighted_crps']}"

    for zone, crps_val in results['zone_breakdown'].items():
        assert crps_val >= 0, f"Zone {zone} CRPS must be >= 0, got {crps_val}"


if __name__ == "__main__":
    print("Glucose-RCRPS Computation Script")
    print("=" * 50)
    print("\nThis script computes Glucose-RCRPS from cached prediction results.")
    print("\nComponents:")
    print("1. Clarke zone-weighted CRPS (RoI weighting)")
    print("2. Constraint penalties (rate mismatch + bounds)")
    print("3. Scale normalization (α = 1/110)")
    print("\nFormula: Glucose-RCRPS = α · [Clarke-CRPS + β · Constraint-CRPS]")
    print(f"         where α = {DEFAULT_ALPHA:.6f}, β = {DEFAULT_BETA}")
    print("\nTo use this module, import and call compute_glucose_rcrps()")
    print("See docstrings for detailed usage.")
