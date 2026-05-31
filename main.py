from time import sleep
from backtest import run_all_backtests


def format_series(series):
    return series.to_string(float_format=lambda x: f"{x:,.2f}")


def print_per_symbol_summary(per_symbol):
    for strategy_name, data in per_symbol.items():
        print(f"\n--- Strategy: {strategy_name} ---")
        print("Portfolio summary:")
        print(format_series(data["summary"]))
        print("\nPer-symbol stats:")
        print(data["stats"].to_string(float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    pf, stats, per_symbol, price_df, entries, exits = run_all_backtests()

    print("\n=== Strategy Comparison ===")
    print(stats)
    sleep(2)  # ensure stats print before per-symbol summary
    print("\n=== Per-symbol summary ===")
    print_per_symbol_summary(per_symbol)
