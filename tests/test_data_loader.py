"""Tests for the data pipeline (network-free)."""

import numpy as np

from syntheticmarket.data.loader import make_sliding_windows, scale_prices


def test_sliding_window_shape():
    series = np.arange(100, dtype="float32")
    windows = make_sliding_windows(series, seq_len=24, step=1)
    assert windows.shape == (100 - 24 + 1, 24, 1)


def test_scaled_values_in_unit_range():
    prices = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype="float32")
    scaled, _ = scale_prices(prices)
    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0
    # endpoints map exactly to the range bounds
    assert np.isclose(scaled.min(), 0.0)
    assert np.isclose(scaled.max(), 1.0)


def test_scaler_is_deterministic():
    prices = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], dtype="float32")
    scaled_a, _ = scale_prices(prices)
    scaled_b, _ = scale_prices(prices)
    np.testing.assert_array_equal(scaled_a, scaled_b)


def test_sliding_window_values_match_source():
    series = np.arange(30, dtype="float32")
    windows = make_sliding_windows(series, seq_len=24, step=1)
    # first window is the first 24 values
    np.testing.assert_array_equal(windows[0, :, 0], series[:24])
