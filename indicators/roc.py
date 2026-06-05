import pandas as pd
import pandas_ta as ta

def add_roc(df: pd.DataFrame, length: int = 12) -> pd.DataFrame:
    """Add Rate of Change (ROC) column to DataFrame.

    Args:
        df: DataFrame with 'close' column.
        length: lookback period for ROC (default 12).

    Returns:
        DataFrame with 'roc' column added.
    """
    df["roc"] = ta.roc(df["close"], length=length)
    return df
