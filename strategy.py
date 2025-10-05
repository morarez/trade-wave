# strategy.py
from indicators import compute_indicators, generate_signals_from_df

def generate_signals(data):
    """
    Generate buy/sell/hold signals for all tickers (live mode).
    """
    indicators = compute_indicators(data)
    signals = {}
    for symbol, df in indicators.items():
        signal, latest = generate_signals_from_df(df)
        signals[symbol] = {
            "signal": signal,
            "price": latest["close"],
            "rsi": latest["rsi"],
            "sma_short": latest["sma_short"],
            "sma_long": latest["sma_long"]
        }
    return signals
