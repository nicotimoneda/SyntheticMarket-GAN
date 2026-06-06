"""Data loading and preprocessing for AAPL price windows."""

from syntheticmarket.data.loader import (
    DEFAULT_END,
    DEFAULT_START,
    DEFAULT_TICKER,
    SEQ_LEN,
    download_close_prices,
    load_price_windows,
    make_sliding_windows,
    scale_prices,
)

__all__ = [
    "DEFAULT_TICKER",
    "DEFAULT_START",
    "DEFAULT_END",
    "SEQ_LEN",
    "download_close_prices",
    "scale_prices",
    "make_sliding_windows",
    "load_price_windows",
]
