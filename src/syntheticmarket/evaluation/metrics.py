"""Temporal-fidelity metrics.

The headline metric from the blog post is the **step-to-step standard deviation**:
the standard deviation of day-to-day differences within the price windows. The
published values are 0.009 (real) vs 0.039 (synthetic), a ratio of ~4.3x.
"""

from __future__ import annotations

import numpy as np


def _as_2d(windows: np.ndarray) -> np.ndarray:
    """Coerce ``(N, T, 1)`` or ``(N, T)`` windows to ``(N, T)``."""
    arr = np.asarray(windows, dtype="float64")
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def step_to_step_std(windows: np.ndarray) -> float:
    """Std of all consecutive day-to-day differences across the windows.

    Differences are taken along the time axis of each window and pooled before
    computing a single standard deviation. A constant series yields 0.0.
    """
    arr = _as_2d(windows)
    diffs = np.diff(arr, axis=1)
    return float(diffs.std())


def step_to_step_ratio(real: np.ndarray, synthetic: np.ndarray) -> float:
    """Ratio of synthetic to real step-to-step std (``inf`` if real std is 0)."""
    real_std = step_to_step_std(real)
    synth_std = step_to_step_std(synthetic)
    if real_std == 0.0:
        return float("inf")
    return synth_std / real_std


def summary_metrics(real: np.ndarray, synthetic: np.ndarray) -> dict[str, float]:
    """Dict of the reported temporal-fidelity metrics for real vs synthetic."""
    real_std = step_to_step_std(real)
    synth_std = step_to_step_std(synthetic)
    return {
        "step_to_step_std_real": real_std,
        "step_to_step_std_synthetic": synth_std,
        "step_to_step_ratio": (synth_std / real_std if real_std else float("inf")),
    }
