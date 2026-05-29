import pandas as pd
from indicators.moving_averages import add_sma
from indicators.momentum import add_rsi

RSI_LOWER = 30
RSI_UPPER = 70

def apply_indicators(df):
    df = add_sma(df)
    df = add_rsi(df)
    return df


def generate_signals(df):
    df = apply_indicators(df)
    signals = pd.Series("HOLD", index=df.index)

    sma_short = pd.to_numeric(df["sma_short"], errors="coerce")
    sma_long = pd.to_numeric(df["sma_long"], errors="coerce")
    rsi = pd.to_numeric(df["rsi"], errors="coerce")

    buy = (sma_short > sma_long) & (rsi > RSI_LOWER) & (rsi < RSI_UPPER)
    sell = sma_short < sma_long
    buy = buy.fillna(False)
    sell = sell.fillna(False)

    signals.loc[buy] = "BUY"
    signals.loc[sell] = "SELL"
    return signals


def generate_signal(df):
    latest = apply_indicators(df).iloc[-1]
    if latest["sma_short"] > latest["sma_long"] and RSI_LOWER < latest["rsi"] < RSI_UPPER:
        return "BUY"
    elif latest["sma_short"] < latest["sma_long"]:
        return "SELL"
    else:
        return "HOLD"
