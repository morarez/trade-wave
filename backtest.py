# backtest.py
import vectorbt as vbt
import pandas as pd
from datetime import datetime
from data_handler import get_alpaca_data
from config import TICKERS
from indicators import compute_indicators_vbt, build_signals_vbt

FEES = 0.001
SLIPPAGE = 0.0005

def run_backtest(price_df: pd.DataFrame, entries: pd.DataFrame, exits: pd.DataFrame, cash: float = 10000.0):
    """Run a vectorbt backtest portfolio."""
    pf = vbt.Portfolio.from_signals(
        close=price_df,
        entries=entries,
        exits=exits,
        init_cash=cash,
        fees=FEES,
        slippage=SLIPPAGE,
        freq='1D',
        size=1.0,
        allow_short=False
    )
    return pf


def summarize_portfolio(pf: vbt.Portfolio):
    """Return portfolio stats."""
    stats = pf.stats()
    per_symbol = pd.DataFrame({
        "total_return": pf.total_return(),
        "cagr": pf.cagr(),
        "max_drawdown": pf.max_drawdown(),
        "sharpe": pf.sharpe_ratio()
    })
    return stats, per_symbol


def full_backtest_run(start_date: str = None, cash: float = 10000.0):
    """Run a full backtest using Alpaca data."""
    print("Loading price data from Alpaca...")
    price_df = get_alpaca_data()
    if start_date:
        price_df = price_df[price_df.index >= pd.to_datetime(start_date)]
    price_df = price_df.loc[:, price_df.columns.isin(TICKERS)]

    print("Computing indicators (vectorbt)...")
    fast_ma, slow_ma, rsi_df = compute_indicators_vbt(price_df)

    print("Building entry/exit signals...")
    entries, exits = build_signals_vbt(price_df, fast_ma, slow_ma, rsi_df)

    print("Running vectorbt backtest...")
    pf = run_backtest(price_df, entries, exits, cash=cash)

    stats, per_symbol = summarize_portfolio(pf)
    return pf, stats, per_symbol, price_df, entries, exits
