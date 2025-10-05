import os
from dotenv import load_dotenv

load_dotenv()

# Alpaca configuration
APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
APCA_API_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
APCA_DATA_URL = "https://data.alpaca.markets/v2"

# Strategy parameters
TICKERS = ["AAPL", "MSFT", "NVDA"]
SHORT_WINDOW = 50
LONG_WINDOW = 200
RSI_PERIOD = 14

# Data timeframe
START_DATE = "2020-01-01"
TIMEFRAME = "1Day"
