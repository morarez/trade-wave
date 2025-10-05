from indicators.momentum import add_rsi
from indicators.volatility import add_bollinger_bands

def apply_indicators(df):
    df = add_bollinger_bands(df)
    df = add_rsi(df)
    return df

def generate_signal(df):
    df = apply_indicators(df)
    latest = df.iloc[-1]
    if latest["close"] < latest["BBL_20_2.0"] and latest["rsi"] < 35:
        return "BUY"
    elif latest["close"] > latest["BBU_20_2.0"] and latest["rsi"] > 65:
        return "SELL"
    else:
        return "HOLD"
