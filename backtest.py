# backtest.py
import vectorbt as vbt
from data_handler import get_yfinance_data
from strategies.strategy_factory import STRATEGY_MAP
import pandas as pd
import numpy as np

def run_all_backtests(symbols=None, start_date="2020-01-01", end_date=None, cash=10000, interval="1d"):
    """
    Run backtests for all strategies on given symbols.
    
    Parameters:
        symbols: list of tickers
        start_date: str, start of historical data
        end_date: str, end of historical data (defaults to today)
        cash: initial cash per portfolio
        interval: yfinance interval ('1d', '1m', '5m', etc.)
    
    Returns:
        pf: last portfolio object
        summary: DataFrame of stats per strategy
        per_symbol: dict of {strategy_name: {'pf': Portfolio, 'stats': stats}}
        price_df: DataFrame of price data
        entries: DataFrame of entry signals
        exits: DataFrame of exit signals
    """

    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOG"]

    # Fetch price data from yfinance
    price_df = get_yfinance_data(symbols=symbols, start=start_date, end=end_date, interval=interval)
    price_df = price_df.astype(float).copy()  # ensure float & writeable

    summary = {}
    per_symbol = {}

    for name, strat in STRATEGY_MAP.items():
        print(f"Running backtest for strategy: {name}")

        # Prepare empty boolean DataFrames
        entries = pd.DataFrame(False, index=price_df.index, columns=price_df.columns, dtype=bool)
        exits = pd.DataFrame(False, index=price_df.index, columns=price_df.columns, dtype=bool)

        # Generate signals for each symbol
        for symbol in price_df.columns:
            df = price_df[[symbol]].rename(columns={symbol: "close"})
            signal = strat.generate_signal(df)  # should return Series of "BUY"/"SELL"/"HOLD" or single value

            # If single value, broadcast to all dates
            if isinstance(signal, str) or isinstance(signal, bool):
                signal_value = signal == "BUY"
                entries[symbol] = pd.Series(signal_value, index=price_df.index, dtype=bool)
                exits[symbol] = pd.Series(~signal_value, index=price_df.index, dtype=bool)
            else:
                # Series: convert to bool
                entries[symbol] = (signal == "BUY").astype(bool)
                exits[symbol] = (signal == "SELL").astype(bool)

        # Convert to NumPy arrays and make writeable to avoid read-only errors
        # Make sure entries/exits are boolean and no NaNs
        entries = entries.fillna(False).astype(bool)
        exits = exits.fillna(False).astype(bool)
        price_df = price_df.copy()

        pf = vbt.Portfolio.from_signals(
            price_df,
            entries,
            exits,
            init_cash=cash,
            freq="1D"
        )

        stats = pf.stats(metrics=["total_return", "sharpe_ratio"])
        summary[name] = stats[['Total Return [%]', 'Sharpe Ratio']]
        per_symbol[name] = {"pf": pf, "stats": stats}

    result_summary = pd.DataFrame(summary).T
    print("\n=== Strategy Comparison ===")
    print(result_summary)

    return pf, result_summary, per_symbol, price_df, entries, exits
