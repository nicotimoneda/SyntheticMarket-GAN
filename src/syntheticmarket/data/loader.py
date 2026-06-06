"""AAPL data pipeline: yfinance download -> MinMax[0, 1] -> sliding windows.

Mirrors the preprocessing described in the blog post (Section 3): univariate
Close price, MinMax scaling on the full series, sliding window of length 24 with
step 1. The pure functions (:func:`scale_prices`, :func:`make_sliding_windows`)
are network-free and unit-tested; only :func:`download_close_prices` touches the
network.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Paper defaults (do not change).
DEFAULT_TICKER = "AAPL"
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2025-11-29"
SEQ_LEN = 24
STEP = 1
FEATURE_RANGE = (0.0, 1.0)


def download_close_prices(
    ticker: str = DEFAULT_TICKER,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> np.ndarray:
    """Download daily Close prices from Yahoo Finance as a 1-D float array."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker} ({start} -> {end}).")
    close = df["Close"]
    return np.asarray(close, dtype="float32").reshape(-1)


def scale_prices(
    prices: np.ndarray,
    feature_range: tuple[float, float] = FEATURE_RANGE,
) -> tuple[np.ndarray, MinMaxScaler]:
    """MinMax-scale a 1-D price array to ``feature_range``.

    Returns the scaled 1-D array and the fitted scaler (for inverse transforms).
    """
    prices = np.asarray(prices, dtype="float32").reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=feature_range)
    scaled = scaler.fit_transform(prices).reshape(-1)
    return scaled.astype("float32"), scaler


def make_sliding_windows(
    series: np.ndarray,
    seq_len: int = SEQ_LEN,
    step: int = STEP,
) -> np.ndarray:
    """Turn a 1-D series into overlapping windows of shape ``(N, seq_len, 1)``."""
    series = np.asarray(series, dtype="float32").reshape(-1)
    if series.shape[0] < seq_len:
        raise ValueError(f"Series length {series.shape[0]} shorter than seq_len {seq_len}.")
    windows = [series[i : i + seq_len] for i in range(0, len(series) - seq_len + 1, step)]
    arr = np.stack(windows).astype("float32")
    return arr[:, :, None]  # (N, seq_len, 1)


def load_price_windows(
    ticker: str = DEFAULT_TICKER,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    seq_len: int = SEQ_LEN,
    step: int = STEP,
) -> tuple[np.ndarray, MinMaxScaler]:
    """End-to-end: download -> scale -> window. Returns ``(windows, scaler)``."""
    prices = download_close_prices(ticker, start, end)
    scaled, scaler = scale_prices(prices)
    windows = make_sliding_windows(scaled, seq_len=seq_len, step=step)
    return windows, scaler
