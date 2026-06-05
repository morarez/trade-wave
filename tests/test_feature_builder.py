import pandas as pd
import numpy as np

from ai.data import features_for_series


def test_features_from_close_series():
    # simple increasing close series
    idx = pd.date_range("2021-01-01", periods=100, freq="D")
    close = pd.Series(np.linspace(50, 100, 100), index=idx)
    feats = features_for_series(close)
    # basic returns present
    assert "ret_1" in feats.columns
    assert "ret_5" in feats.columns
    # RSI should be present (best-effort)
    assert "rsi" in feats.columns


def test_features_from_ohlcv_dataframe():
    idx = pd.date_range("2021-01-01", periods=100, freq="D")
    close = np.linspace(50, 100, 100) + np.random.normal(0, 0.1, 100)
    high = close + np.random.uniform(0.1, 1.0, 100)
    low = close - np.random.uniform(0.1, 1.0, 100)
    volume = np.random.randint(100, 1000, 100)
    df = pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume}, index=idx)
    feats = features_for_series(df)
    # ensures enriched features included
    assert "sma_short" in feats.columns or "ema_short" in feats.columns
    assert "obv" in feats.columns or "vol_change" in feats.columns
