import logging
import time
import pandas as pd
import schedule
from utils import fetch_binance_data, save_signal
from strategies.ma_crossover import generate_signal as ma_signal
from strategies.rsi_strategy import generate_signal as rsi_signal

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("data/trading_bot.log"),
        logging.StreamHandler()
    ]
)

symbols = ["BTCUSDT", "ETHUSDT"]
interval = "5m"
limit = 50

def fetch_and_signal():
    for symbol in symbols:
        try:
            df = fetch_binance_data(symbol, interval, limit)
            
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

# Run once immediately
fetch_and_signal()

# Schedule every 5 minutes
schedule.every(5).minutes.do(fetch_and_signal)

logging.info("Starting 5-minute trading signal bot...")

while True:
    schedule.run_pending()
    time.sleep(1)
