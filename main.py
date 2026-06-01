import argparse
from time import sleep
from backtest import run_all_backtests


def format_series(series):
    """Format a pandas Series with 2-decimal comma-separated numbers.
    
    Args:
        series: pandas Series to format.
    
    Returns:
        String representation of the series with comma-separated values.
    """
    return series.to_string(float_format=lambda x: f"{x:,.2f}")


def print_per_symbol_summary(per_symbol):
    """Print formatted per-symbol and per-strategy performance summaries.
    
    Args:
        per_symbol: dict of {strategy_name: {'summary': pd.Series, 'stats': pd.DataFrame}}.
    """
    for strategy_name, data in per_symbol.items():
        print(f"\n--- Strategy: {strategy_name} ---")
        print("Portfolio summary:")
        print(format_series(data["summary"]))
        print("\nPer-symbol stats:")
        print(data["stats"].to_string(float_format=lambda x: f"{x:,.2f}"))


def parse_symbol_list(symbol_string):
    """Parse a comma/space-separated symbol string into a list."""
    if not symbol_string:
        return None
    symbols = [part.strip().upper() for part in symbol_string.replace(",", " ").split() if part.strip()]
    return symbols or None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run strategy backtests for a list of symbols.")
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma or space separated list of symbols to backtest. Example: AAPL,MSFT,GOOG",
    )
    args = parser.parse_args()

    symbols = parse_symbol_list(args.symbols)
    pf, stats, per_symbol, price_df, entries, exits = run_all_backtests(symbols=symbols)

    print("\n=== Per-symbol summary ===")
    print_per_symbol_summary(per_symbol)
