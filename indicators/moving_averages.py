import pandas as pd
import pandas_ta as ta

def add_sma(df: pd.DataFrame, short_window: int = 20, long_window: int = 50):
    """Add simple moving average columns to DataFrame.
    
    Computes and adds 'sma_short' and 'sma_long' columns to the input DataFrame.
    
    Args:
        df: DataFrame with 'close' column.
        short_window: lookback period for short SMA (default 20).
        long_window: lookback period for long SMA (default 50).
    
    Returns:
        DataFrame with sma_short and sma_long columns added.
    """
    df["sma_short"] = ta.sma(df["close"], length=short_window)
    df["sma_long"] = ta.sma(df["close"], length=long_window)
    return df

def add_ema(df: pd.DataFrame, short_window: int = 12, long_window: int = 26):
    """Add exponential moving average columns to DataFrame.
    
    Computes and adds 'ema_short' and 'ema_long' columns to the input DataFrame.
    
    Args:
        df: DataFrame with 'close' column.
        short_window: lookback period for short EMA (default 12).
        long_window: lookback period for long EMA (default 26).
    
    Returns:
        DataFrame with ema_short and ema_long columns added.
    """
    df["ema_short"] = ta.ema(df["close"], length=short_window)
    df["ema_long"] = ta.ema(df["close"], length=long_window)
    return df

def add_macd(df: pd.DataFrame):
    """Add MACD indicator columns to DataFrame.
    
    Computes and adds MACD line, signal line, and histogram columns.
    
    Args:
        df: DataFrame with 'close' column.
    
    Returns:
        DataFrame with MACD columns added; unchanged if MACD computation fails.
    """
    macd = ta.macd(df["close"])
    if macd is None:
        return df
    df = pd.concat([df, macd], axis=1)
    return df
