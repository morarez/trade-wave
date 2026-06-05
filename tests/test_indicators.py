import pandas as pd
import numpy as np

from indicators import (
    add_atr,
    add_roc,
    add_adx,
    add_obv,
    add_volume_features,
    add_bollinger_bands,
    add_rolling_volatility,
)


def make_ohlcv(n=60):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    closes = np.linspace(100, 120, n) + np.random.normal(0, 0.5, n)
    highs = closes + np.random.uniform(0.1, 1.0, n)
    lows = closes - np.random.uniform(0.1, 1.0, n)
    volume = np.random.randint(100, 1000, n)
    return pd.DataFrame({"high": highs, "low": lows, "close": closes, "volume": volume}, index=idx)


def test_volume_and_obv_and_vwap_and_atr_and_roc():
    df = make_ohlcv()
    df2 = add_volume_features(df.copy())
    assert "vol_change" in df2.columns
    assert "vol_ema_20" in df2.columns
    assert "vwap" in df2.columns

    df3 = add_obv(df.copy())
    assert "obv" in df3.columns

    df4 = add_atr(df.copy())
    assert "atr" in df4.columns

    df5 = add_roc(df.copy())
    assert "roc" in df5.columns


def test_adx_and_bollinger_and_volatility():
    df = make_ohlcv()
    df_adx = add_adx(df.copy())
    # pandas_ta adx returns ADX_, DMP_, DMN_ columns with suffix length
    assert any(c.startswith("ADX") for c in df_adx.columns)
    assert any(c.startswith("DMP") or c.startswith("DMN") for c in df_adx.columns)

    df_bb = add_bollinger_bands(df.copy())
    assert any("BBL" in c or "BBM" in c or "BBU" in c for c in df_bb.columns)

    df_vol = add_rolling_volatility(df.copy())
    assert "volatility" in df_vol.columns
