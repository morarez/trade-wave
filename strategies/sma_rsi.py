from indicators.moving_averages import add_sma
from indicators.momentum import add_rsi

RSI_LOWER = 30
RSI_UPPER = 70

def apply_indicators(df):
    df = add_sma(df)
    df = add_rsi(df)
    return df

def generate_signal(df):
    df = apply_indicators(df)
    latest = df.iloc[-1]
    if latest["sma_short"] > latest["sma_long"] and RSI_LOWER < latest["rsi"] < RSI_UPPER:
        return "BUY"
    elif latest["sma_short"] < latest["sma_long"]:
        return "SELL"
    else:
        return "HOLD"
