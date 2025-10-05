# main_backtest.py
from backtest import run_all_backtests
import matplotlib.pyplot as plt

if __name__ == "__main__":
    pf, stats, per_symbol, price_df, entries, exits = run_all_backtests()

    print("=== Overall Stats ===")
    print(stats)
    print("\n=== Per-symbol summary ===")
    print(per_symbol)

    pf.plot().show()
