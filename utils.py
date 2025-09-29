import requests
import pandas as pd
import os

def fetch_binance_data(symbol="BTCUSDT", interval="1h", limit=100):
    """Fetch historical candle data from Binance"""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","num_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df['close'] = pd.to_numeric(df['close'])
    return df

def save_signal(symbol, signal, filename="data/signals.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df_new = pd.DataFrame([{"symbol": symbol, "signal": signal}])
    
    if os.path.exists(filename):
        df_new.to_csv(filename, mode='a', header=False, index=False)
    else:
        df_new.to_csv(filename, index=False)

