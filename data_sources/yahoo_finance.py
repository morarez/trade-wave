import yfinance as yf
import pandas as pd

def fetch_yfinance_data(symbols="AAPL", interval="1h", period="7d"):
    """
    Fetch historical candle data from Yahoo Finance (NASDAQ stocks)
    
    symbols: str or list - single ticker or list of tickers
    interval: str - "1m","5m","15m","30m","60m","90m","1h","1d","1wk","1mo","3mo"
    period: str - "1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"
    """
    df = yf.download(
        tickers=symbols,
        interval=interval,
        period=period,
        auto_adjust=False
    )

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col).strip() for col in df.columns.values]

    df = df.reset_index()

    # Rename for consistency with single-ticker structure
    rename_map = {}
    for col in df.columns:
        if col.lower().startswith("open"):
            rename_map[col] = "open"
        elif col.lower().startswith("high"):
            rename_map[col] = "high"
        elif col.lower().startswith("low"):
            rename_map[col] = "low"
        elif col.lower().startswith("close"):
            rename_map[col] = "close"
        elif col.lower().startswith("adjclose"):
            rename_map[col] = "adj_close"
        elif col.lower().startswith("volume"):
            rename_map[col] = "volume"
        elif col.lower() in ["date", "datetime"]:
            rename_map[col] = "open_time"

    df = df.rename(columns=rename_map)
    return df
