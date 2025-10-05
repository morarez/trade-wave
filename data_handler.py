# data_handler.py
import yfinance as yf
import pandas as pd
from typing import List
from datetime import datetime, timedelta

def get_yfinance_data(
    symbols: List[str],
    start: str = "2020-01-01",
    end: str = None,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch historical data from yfinance.

    symbols : list of stock tickers
    start : start date (YYYY-MM-DD)
    end : end date (YYYY-MM-DD), defaults to today
    interval : '1d', '1m', '5m', '15m', '1h', etc.

    Returns a multi-column DataFrame: columns = tickers, rows = datetime index
    """

    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    all_data = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)
        if df.empty:
            print(f"Warning: No data for {symbol}")
            continue
        all_data[symbol] = df['Close']  # keep only close prices

    # Combine into a single DataFrame
    price_df = pd.concat(all_data, axis=1)
    price_df.columns.name = None
    return price_df
