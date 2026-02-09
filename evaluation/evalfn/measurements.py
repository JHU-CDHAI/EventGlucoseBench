"""Minimal measurement utilities for evalfn package.

Provides helpers to load cached prediction artifacts produced by the
EventGlucose evaluation pipelines without depending on external packages.
"""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any, Dict


def load_complete_data(pkl_path: Path | str) -> Dict[str, Any]:
    """Load a complete_data.pkl artifact produced by the pipelines.

    The expected structure is a dict with keys like:
      - 'predictions': {'samples': np.ndarray, ...}
      - 'input_data': {'future_time': np.ndarray or pandas structure}
    """
    p = Path(pkl_path)
    with p.open('rb') as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid complete_data format at {p}: expected dict, got {type(data)}")
    return data


def n_samples_from_complete_data(data: Dict[str, Any]) -> int:
    """Return the number of predictive samples contained in complete_data."""
    samples = data.get('predictions', {}).get('samples')
    if samples is None:
        return 0
    try:
        return int(getattr(samples, 'shape', [len(samples)])[0])
    except Exception:
        return 0

