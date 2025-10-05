# indicators.py
import pandas as pd
import pandas_ta as ta
import vectorbt as vbt
from config import SHORT_WINDOW, LONG_WINDOW, RSI_PERIOD

RSI_LOWER = 30
RSI_UPPER = 70

def compute_indicators(data: pd.DataFrame):
    """
    Compute SMA and RSI indicators for each ticker using pandas-ta (for live mode).
    Returns a dict of DataFrames keyed by ticker.
    """
    indicators = {}
    for symbol in data.columns:
        df = pd.DataFrame()
        df["close"] = data[symbol]
        df["sma_short"] = ta.sma(df["close"], length=SHORT_WINDOW)
        df["sma_long"] = ta.sma(df["close"], length=LONG_WINDOW)
        df["rsi"] = ta.rsi(df["close"], length=RSI_PERIOD)
        indicators[symbol] = df
    return indicators


def compute_indicators_vbt(data: pd.DataFrame):
    """
    Compute SMA and RSI using vectorbt runners (for backtesting).
    Returns fast_ma, slow_ma, rsi DataFrames aligned with `data`.
    """
    fast = vbt.MA.run(data, window=SHORT_WINDOW)
    slow = vbt.MA.run(data, window=LONG_WINDOW)
    rsi = vbt.RSI.run(data, window=RSI_PERIOD)
    return fast.ma, slow.ma, rsi.rsi


def generate_signals_from_df(df: pd.DataFrame):
    """
    Generate a single signal for one ticker's DataFrame (used in live mode).
    """
    buy = (
        (df["sma_short"] > df["sma_long"]) &
        (df["rsi"] > RSI_LOWER) &
        (df["rsi"] < RSI_UPPER)
    )
    sell = df["sma_short"] < df["sma_long"]
    latest = df.iloc[-1]
    signal = "BUY" if buy.iloc[-1] else "SELL" if sell.iloc[-1] else "HOLD"
    return signal, latest


def build_signals_vbt(price_df: pd.DataFrame, fast_ma: pd.DataFrame, slow_ma: pd.DataFrame, rsi_df: pd.DataFrame):
    """
    Build vectorized boolean entry/exit signals (for backtesting).
    """
    entries = (fast_ma > slow_ma) & (rsi_df > RSI_LOWER) & (rsi_df < RSI_UPPER)
    exits = (fast_ma < slow_ma)
    entries = entries.reindex_like(price_df).fillna(False)
    exits = exits.reindex_like(price_df).fillna(False)
    return entries, exits
