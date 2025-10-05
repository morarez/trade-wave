# signal_service.py
from data_handler import get_alpaca_data
from strategy import generate_signals
from utils import log_signal

def run_signal_service():
    """Fetch data, compute indicators, and generate signals."""
    print("Fetching latest data...")
    prices = get_alpaca_data()

    print("Generating signals...")
    signals = generate_signals(prices)

    for symbol, info in signals.items():
        log_signal(symbol, info)
    return signals
