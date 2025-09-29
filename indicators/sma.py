def sma(df, window=20):
    """Simple Moving Average"""
    return df['close'].rolling(window).mean()
