import pandas as pd
import pandas_ta as ta

def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Add On-Balance Volume (OBV) column to DataFrame.

    Args:
        df: DataFrame with 'close' and 'volume' columns.

    Returns:
        DataFrame with 'obv' column added.
    """
    if "volume" not in df.columns:
        return df
    df["obv"] = ta.obv(df["close"], df["volume"])
    return df
