# backtest.py
import vectorbt as vbt
from data_handler import get_yfinance_data
from strategies.strategy_factory import STRATEGY_MAP
import pandas as pd
import numpy as np

def run_all_backtests(symbols=None, start_date="2024-01-01", end_date=None, cash=10000, interval="1d"):
    """
    Run backtests for all strategies on given symbols.
    
    Parameters:
        symbols: list of tickers
        start_date: str, start of historical data
        end_date: str, end of historical data (defaults to today)
        cash: initial cash per portfolio
        interval: yfinance interval ('1d', '1m', '5m', etc.)
    
    Returns:
        portfolios: dict of {strategy_name: Portfolio} for every backtest run
        summary: DataFrame of stats per strategy
        per_symbol: dict of {strategy_name: {'summary': pd.Series, 'stats': pd.DataFrame}}
        price_df: DataFrame of price data
        entries: DataFrame of entry signals
        exits: DataFrame of exit signals
    """

    if symbols is None:
        symbols = ["AAPL", "SOFI", "GOOG"]

    # Fetch price data from yfinance
    price_df = get_yfinance_data(symbols=symbols, start=start_date, end=end_date, interval=interval)
    price_df = price_df.astype(float).copy()  # ensure float & writeable

    summary = {}
    per_symbol = {}
    portfolios = {}

    def build_entry_exit_signals(signal, index):
        """Convert BUY/SELL/HOLD signals into vectorbt entry/exit boolean Series.
        
        Detects signal transitions: BUY becomes an entry (first occurrence after non-BUY),
        SELL becomes an exit (first occurrence after non-SELL). This prevents consecutive 
        duplicate signals and produces clean entry/exit points for portfolio construction.
        
        Args:
            signal: either a pd.Series of signals ('BUY', 'SELL', 'HOLD') indexed by datetime,
                    or a single string signal ('BUY', 'SELL', 'HOLD').
            index: the target datetime index to align signals to (typically price_df.index).
        
        Returns:
            Tuple of (entries, exits) where each is a pd.Series of bool indexed by the 
            provided index. True indicates a transition into that signal state.
        """
        if isinstance(signal, pd.Series):
            signal = signal.reindex(index).astype("string").fillna("HOLD")
            buy = signal == "BUY"
            sell = signal == "SELL"
            entries = buy & ~buy.shift(1).fillna(False)
            exits = sell & ~sell.shift(1).fillna(False)
            return entries.astype(bool), exits.astype(bool)

        entries = pd.Series(False, index=index, dtype=bool)
        exits = pd.Series(False, index=index, dtype=bool)
        if signal == "BUY" and len(index) > 0:
            entries.iloc[0] = True
        return entries, exits

    for name, strat in STRATEGY_MAP.items():
        print(f"Running backtest for strategy: {name}")

        entries = pd.DataFrame(False, index=price_df.index, columns=price_df.columns, dtype=bool)
        exits = pd.DataFrame(False, index=price_df.index, columns=price_df.columns, dtype=bool)

        for symbol in price_df.columns:
            df = price_df[[symbol]].rename(columns={symbol: "close"})
            if hasattr(strat, "generate_signals"):
                signal = strat.generate_signals(df)
            else:
                signal = strat.generate_signal(df)

            symbol_entries, symbol_exits = build_entry_exit_signals(signal, price_df.index)
            entries[symbol] = symbol_entries
            exits[symbol] = symbol_exits

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

        stats = pf.stats(metrics=["total_return", "sharpe_ratio"], agg_func=None)
        pf_summary = stats[['Total Return [%]', 'Sharpe Ratio']].mean()
        summary[name] = pf_summary
        per_symbol[name] = {"summary": pf_summary, "stats": stats}
        portfolios[name] = pf

    result_summary = pd.DataFrame(summary).T

    return portfolios, result_summary, per_symbol, price_df, entries, exits
