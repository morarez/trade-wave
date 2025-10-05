# signal_service.py
from data_handler import get_yfinance_data
from strategies.strategy_factory import get_strategy
from utils import log_signal

def run_signal_service(strategy_name="sma_rsi", symbols=None):
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOG"]

    strategy = get_strategy(strategy_name)
    if strategy is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    print(f"Running {strategy_name} strategy...")
    prices = get_yfinance_data(symbols=symbols, interval="1d")

    signals = {}
    for symbol in prices.columns:
        df = prices[[symbol]].rename(columns={symbol: "close"})
        signal = strategy.generate_signal(df)
        signals[symbol] = signal
        log_signal(symbol, {"strategy": strategy_name, "signal": signal})

    return signals
