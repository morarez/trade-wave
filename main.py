# main_backtest.py
from backtest import full_backtest_run
import matplotlib.pyplot as plt

if __name__ == "__main__":
    pf, stats, per_symbol, price_df, entries, exits = full_backtest_run(start_date="2018-01-01", cash=100000)

    print("=== Overall Stats ===")
    print(stats)
    print("\n=== Per-symbol summary ===")
    print(per_symbol)

    pf.plot().show()
