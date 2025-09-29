def generate_signal(df, short_window=20, long_window=50):
    df['SMA_short'] = df['close'].rolling(short_window).mean()
    df['SMA_long'] = df['close'].rolling(long_window).mean()

    if df['SMA_short'].iloc[-1] > df['SMA_long'].iloc[-1]:
        return "BUY"
    elif df['SMA_short'].iloc[-1] < df['SMA_long'].iloc[-1]:
        return "SELL"
    else:
        return "HOLD"
