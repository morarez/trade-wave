# strategies/bollinger_rsi.py
import pandas as pd
from indicators.momentum import add_rsi
from indicators.volatility import add_bollinger_bands

def apply_indicators(df):
    """Add Bollinger Bands and RSI columns."""
    df = add_bollinger_bands(df)
    df = add_rsi(df)
    return df

def generate_signal(df):
    """
    Generate BUY / SELL / HOLD signal using Bollinger Bands + RSI.
    Ensures indicator columns exist even if helper functions use different names.
    """
    df = apply_indicators(df)
    latest = df.iloc[-1]

    # --- Normalize possible column names ---
    # Try to detect Bollinger Bands column naming dynamically
    bbl_col = next((c for c in df.columns if c.lower().startswith("bbl")), None)
    bbu_col = next((c for c in df.columns if c.lower().startswith("bbu")), None)
    rsi_col = next((c for c in df.columns if c.lower() == "rsi"), None)

    if bbl_col is None or bbu_col is None or rsi_col is None:
        raise KeyError(
            f"Missing required indicator columns. Found columns: {df.columns.tolist()}"
        )

    # --- Generate signal based on latest values ---
    if latest["close"] < latest[bbl_col] and latest[rsi_col] < 35:
        return "BUY"
    elif latest["close"] > latest[bbu_col] and latest[rsi_col] > 65:
        return "SELL"
    else:
        return "HOLD"
