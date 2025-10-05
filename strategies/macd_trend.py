from indicators.moving_averages import add_macd

def apply_indicators(df):
    df = add_macd(df)
    return df

def generate_signal(df):
    df = apply_indicators(df)
    latest = df.iloc[-1]
    if latest["MACD_12_26_9"] > latest["MACDs_12_26_9"]:
        return "BUY"
    elif latest["MACD_12_26_9"] < latest["MACDs_12_26_9"]:
        return "SELL"
    else:
        return "HOLD"
