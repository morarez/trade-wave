import pandas as pd
import pandas_ta as ta

def add_rsi(df: pd.DataFrame, length: int = 14):
    df["rsi"] = ta.rsi(df["close"], length=length)
    return df

def add_stochastic(df: pd.DataFrame):
    stoch = ta.stoch(df["high"], df["low"], df["close"])
    df = pd.concat([df, stoch], axis=1)
    return df
