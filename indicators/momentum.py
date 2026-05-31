import pandas as pd
import pandas_ta as ta

def add_rsi(df: pd.DataFrame, length: int = 14):
    """Add Relative Strength Index (RSI) column to DataFrame.
    
    Computes and adds 'rsi' column to the input DataFrame.
    
    Args:
        df: DataFrame with 'close' column.
        length: lookback period for RSI calculation (default 14).
    
    Returns:
        DataFrame with rsi column added.
    """
    df["rsi"] = ta.rsi(df["close"], length=length)
    return df

def add_stochastic(df: pd.DataFrame):
    """Add Stochastic Oscillator columns to DataFrame.
    
    Computes and adds Stochastic %K and %D columns.
    
    Args:
        df: DataFrame with 'high', 'low', and 'close' columns.
    
    Returns:
        DataFrame with stochastic oscillator columns added.
    """
    stoch = ta.stoch(df["high"], df["low"], df["close"])
    df = pd.concat([df, stoch], axis=1)
    return df
