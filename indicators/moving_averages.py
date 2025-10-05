import pandas as pd
import pandas_ta as ta

def add_sma(df: pd.DataFrame, short_window: int = 20, long_window: int = 50):
    df["sma_short"] = ta.sma(df["close"], length=short_window)
    df["sma_long"] = ta.sma(df["close"], length=long_window)
    return df

def add_ema(df: pd.DataFrame, short_window: int = 12, long_window: int = 26):
    df["ema_short"] = ta.ema(df["close"], length=short_window)
    df["ema_long"] = ta.ema(df["close"], length=long_window)
    return df

def add_macd(df: pd.DataFrame):
    macd = ta.macd(df["close"])
    df = pd.concat([df, macd], axis=1)
    return df
