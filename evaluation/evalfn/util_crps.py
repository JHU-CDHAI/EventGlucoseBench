"""Utility CRPS implementation for evalfn.

Implements CRPS for empirical samples against a scalar or vector target.

Formula (Gneiting & Raftery, 2007) for empirical distribution:
  CRPS(F, y) = E|X - y| - 0.5 E|X - X'|
where X and X' are i.i.d. samples from F.
"""

from __future__ import annotations

import numpy as np


def crps(target: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Compute CRPS for each timestep using empirical samples.

    Args:
        target: shape (T,) array of ground truth values
        samples: shape (S, T) array of samples (S = number of samples)

    Returns:
        crps_t: shape (T,) array with CRPS at each timestep
    """
    target = np.asarray(target, dtype=float).reshape(-1)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[:, None]

    S, T = samples.shape
    if target.shape[0] != T:
        # Align lengths defensively
        T = min(T, target.shape[0])
        samples = samples[:, :T]
        target = target[:T]

    # E|X - y| term
    # Broadcast target to (S, T)
    abs_diff = np.abs(samples - target[np.newaxis, :])
    term1 = abs_diff.mean(axis=0)  # shape (T,)

    # 0.5 E|X - X'| term — compute efficiently
    # Sort samples per timestep and use known identity for pairwise absolute deviations
    # sum_{i<j} |x_i - x_j| = sum_{k=1}^S (2k - S - 1) x_{(k)} for sorted x
    sorted_samples = np.sort(samples, axis=0)
    k = np.arange(1, S + 1)[:, None]  # shape (S, 1)
    weights = (2 * k - S - 1)  # shape (S, 1)
    sum_pairwise = (weights * sorted_samples).sum(axis=0)  # shape (T,)
    term2 = sum_pairwise / (S * S)  # E|X - X'| / 2 factor applied below

    crps_t = term1 - 0.5 * term2
    return crps_t

