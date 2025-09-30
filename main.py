import logging
import time
import pandas as pd
import schedule
from utils import save_signal,print_preview
from data_sources.binance import fetch_binance_data
from data_sources.yahoo_finance import fetch_yfinance_data
from strategies.ma_crossover import generate_signal as ma_signal
from strategies.rsi_strategy import generate_signal as rsi_signal
from ai.signals import generate_forecast

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("data/trading_bot.log"),
        logging.StreamHandler()
    ]
)

# Configuration
symbols = ["AAPL", "MSFT"]  # add more symbols here
interval = "5m"             # 5-minute candles
limit = "5d"                # number of candles to fetch

def fetch_and_signal():
    for symbol in symbols:
        try:
            df = fetch_yfinance_data(symbol, interval, limit)
            
            if df.empty:
                logging.warning(f"No data fetched for {symbol}")
                continue

            signal_ma = ma_signal(df)
            signal_rsi = rsi_signal(df)
            
            msg = f"{symbol} - MA Signal: {signal_ma}, RSI Signal: {signal_rsi}"
            logging.info(msg)
            
            save_signal(symbol, f"MA: {signal_ma}, RSI: {signal_rsi}")
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")



def main():
    logging.info("Starting 5-minute trading signal bot...")

    # Run once immediately
    fetch_and_signal()

    # Schedule every 5 minutes
    schedule.every(5).minutes.do(fetch_and_signal)

    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    # Yahoo Finance example
    yahoo_df = fetch_yfinance_data("AAPL", interval="1h", period="5d")
    forecast = generate_forecast(yahoo_df, periods=24, freq="H")
    print("=== Yahoo Finance Prophet Forecast (AAPL) ===")
    print_preview(forecast)

    # Binance example
    binance_df = fetch_binance_data("BTCUSDT", interval="1h", limit=500)
    forecast = generate_forecast(binance_df, periods=24, freq="H")
    print("\n=== Binance Prophet Forecast (BTCUSDT) ===")
    print_preview(forecast)