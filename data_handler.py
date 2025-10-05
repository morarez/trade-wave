from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
from datetime import datetime
from config import APCA_API_KEY_ID, APCA_API_SECRET_KEY, TICKERS, TIMEFRAME, START_DATE

def get_alpaca_data():
    """Fetch historical bars from Alpaca."""
    client = StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)

    timeframe = TimeFrame.Day if TIMEFRAME == "1Day" else TimeFrame.Minute
    request = StockBarsRequest(
        symbol_or_symbols=TICKERS,
        timeframe=TimeFrame.Day,
        start=pd.Timestamp(START_DATE, tz="America/New_York"),
        end=pd.Timestamp(datetime.now(), tz="America/New_York")
    )

    bars = client.get_stock_bars(request)
    df = bars.df  # multi-index DataFrame: (symbol, timestamp)
    df = df.reset_index().pivot(index="timestamp", columns="symbol", values="close")
    df = df.dropna(how="all")
    return df
