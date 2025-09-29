from indicators.rsi import rsi

def generate_signal(df, period=14, oversold=30, overbought=70):
    df['RSI'] = rsi(df, period)
    last_rsi = df['RSI'].iloc[-1]
    
    if last_rsi < oversold:
        return "BUY"
    elif last_rsi > overbought:
        return "SELL"
    else:
        return "HOLD"
