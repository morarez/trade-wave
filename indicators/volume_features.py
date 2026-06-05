import pandas as pd
import pandas_ta as ta

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple volume-based features to DataFrame.

    Adds volume percent change, 20-period EMA of volume and VWAP (when available).
    """
    if "volume" not in df.columns:
        return df
    df["vol_change"] = df["volume"].pct_change()
    df["vol_ema_20"] = ta.ema(df["volume"], length=20)
    # VWAP needs high, low, close, volume
    if set(["high", "low", "close"]).issubset(df.columns):
        df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    return df
