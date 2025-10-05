from datetime import datetime

def log_signal(symbol, info):
    """Print or log a formatted signal."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {symbol}: {info['signal']} "
          f"(Price={info['price']:.2f}, RSI={info['rsi']:.1f}, "
          f"SMA50={info['sma_short']:.2f}, SMA200={info['sma_long']:.2f})")
