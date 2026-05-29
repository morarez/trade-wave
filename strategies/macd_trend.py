import pandas as pd
from indicators.moving_averages import add_macd

def apply_indicators(df):
    df = add_macd(df)
    return df


def generate_signals(df):
    df = apply_indicators(df)
    signals = pd.Series("HOLD", index=df.index)

    macd = pd.to_numeric(df["MACD_12_26_9"], errors="coerce")
    macds = pd.to_numeric(df["MACDs_12_26_9"], errors="coerce")

    buy = macd > macds
    sell = macd < macds
    buy = buy.fillna(False)
    sell = sell.fillna(False)

    signals.loc[buy] = "BUY"
    signals.loc[sell] = "SELL"
    return signals


def generate_signal(df):
    df = apply_indicators(df)
    latest = df.iloc[-1]
    if latest["MACD_12_26_9"] > latest["MACDs_12_26_9"]:
        return "BUY"
    elif latest["MACD_12_26_9"] < latest["MACDs_12_26_9"]:
        return "SELL"
    else:
        return "HOLD"
