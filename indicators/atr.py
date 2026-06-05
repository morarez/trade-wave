import pandas as pd
import pandas_ta as ta

def add_atr(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Add Average True Range (ATR) column to DataFrame.

    Args:
        df: DataFrame with 'high', 'low', and 'close' columns.
        length: lookback period for ATR (default 14).

    Returns:
        DataFrame with 'atr' column added.
    """
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=length)
    return df
