import pandas as pd
import pandas_ta as ta

def add_bollinger_bands(df, length=20, std=2):
    """Add Bollinger Bands columns to DataFrame.
    
    Computes and adds upper band, middle band, and lower band columns.
    
    Args:
        df: DataFrame with 'close' column.
        length: lookback period for Bollinger Bands (default 20).
        std: number of standard deviations for band width (default 2).
    
    Returns:
        DataFrame with Bollinger Bands columns added.
    """
    bb = ta.bbands(df["close"], length=length, std=std)
    df = df.join(bb)
    return df

def add_rolling_volatility(df, window: int = 20) -> pd.DataFrame:
    """Add rolling volatility (std of returns) to DataFrame.

    Adds 'volatility' column as rolling std of log returns.
    """
    import numpy as np
    if "close" not in df.columns:
        return df
    df["logret"] = np.log(df["close"]).diff()
    df["volatility"] = df["logret"].rolling(window).std() * (252 ** 0.5)
    df.drop(columns=["logret"], inplace=True)
    return df
