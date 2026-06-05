import pandas as pd
import pandas_ta as ta

def add_adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Add ADX (Average Directional Index) columns to DataFrame.

    Adds 'ADX_', 'DMP_', and 'DMN_' columns produced by pandas_ta.
    """
    adx = ta.adx(df["high"], df["low"], df["close"], length=length)
    if adx is None:
        return df
    df = pd.concat([df, adx], axis=1)
    return df
