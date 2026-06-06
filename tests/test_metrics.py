"""Tests for the step-to-step volatility metric."""

import numpy as np

from syntheticmarket.evaluation.metrics import step_to_step_ratio, step_to_step_std


def test_constant_series_has_zero_step_std():
    windows = np.full((3, 24, 1), 0.5, dtype="float32")
    assert step_to_step_std(windows) == 0.0


def test_known_increments_match_expected_std():
    # One window whose day-to-day differences are exactly [2, -1, 2, -1].
    window = np.array([[0.0, 2.0, 1.0, 3.0, 2.0]], dtype="float64")  # (1, 5)
    expected = np.std(np.diff(window[0]))
    assert np.isclose(step_to_step_std(window), expected)


def test_accepts_both_2d_and_3d_inputs():
    window_2d = np.array([[0.0, 1.0, 0.0, 1.0]], dtype="float64")
    window_3d = window_2d[:, :, None]
    assert np.isclose(step_to_step_std(window_2d), step_to_step_std(window_3d))


def test_ratio_is_synthetic_over_real():
    # real diffs = [1, -1, 1, -1] -> std 1.0 ; synthetic diffs doubled -> std 2.0
    real = np.array([[0.0, 1.0, 0.0, 1.0, 0.0]], dtype="float64")
    synthetic = np.array([[0.0, 2.0, 0.0, 2.0, 0.0]], dtype="float64")
    assert np.isclose(step_to_step_std(real), 1.0)
    assert np.isclose(step_to_step_std(synthetic), 2.0)
    assert np.isclose(step_to_step_ratio(real, synthetic), 2.0)


def test_ratio_is_inf_when_real_is_constant():
    real = np.full((1, 5), 0.5, dtype="float64")  # zero diffs -> std exactly 0
    synthetic = np.array([[0.0, 1.0, 0.0, 1.0, 0.0]], dtype="float64")
    assert step_to_step_std(real) == 0.0
    assert step_to_step_ratio(real, synthetic) == float("inf")
