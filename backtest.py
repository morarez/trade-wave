import vectorbt as vbt
from data_handler import get_yfinance_data
from strategies.strategy_factory import BENCHMARK_STRATEGY_MAP, STRATEGY_MAP
import pandas as pd
import numpy as np
from types import SimpleNamespace
from ai.predict import predict_signals_for_model_path


def _coerce_numeric_series(values):
    numeric = pd.to_numeric(values, errors="coerce")
    if hasattr(numeric, "replace"):
        return numeric.replace([np.inf, -np.inf], np.nan)
    if pd.isna(numeric):
        return np.nan
    return float(numeric)


def _make_ai_strategy(name: str, model_path: str, threshold: float = 0.001):
    def generate_signals(df: pd.DataFrame):
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")
        return predict_signals_for_model_path(df["close"].astype(float).copy(), model_path=model_path, threshold=threshold)

    def generate_signal(df: pd.DataFrame):
        sigs = generate_signals(df)
        return sigs.dropna().iloc[-1] if not sigs.empty else "HOLD"

    return SimpleNamespace(generate_signals=generate_signals, generate_signal=generate_signal)


def run_all_backtests(
    symbols=None,
    start_date="2024-01-01",
    end_date=None,
    cash=10000,
    interval="1d",
    strategy_names=None,
    ai_models=None,
):
    """
    Run backtests for benchmark strategies and optional AI models.

    Parameters:
        symbols: list of tickers
        start_date: str, start of historical data
        end_date: str, end of historical data (defaults to today)
        cash: initial cash per portfolio
        interval: yfinance interval ('1d', '1m', '5m', '15m', '1h', etc.)
        strategy_names: optional list of strategy names to run
        ai_models: optional list of dicts with keys 'name', 'path', and 'threshold'

    Returns:
        portfolios: dict of {strategy_name: Portfolio}
        summary: DataFrame of stats per strategy
        per_symbol: dict of {strategy_name: {'summary': Series, 'stats': DataFrame}}
        price_df: DataFrame of price data
        entries: DataFrame of entry signals for the last strategy processed
        exits: DataFrame of exit signals for the last strategy processed
    """

    if symbols is None:
        symbols = ["AAPL", "SOFI", "GOOG"]

    price_df = get_yfinance_data(symbols=symbols, start=start_date, end=end_date, interval=interval)
    price_df = price_df.astype(float).copy()

    summary = {}
    per_symbol = {}
    portfolios = {}

    observation_period_days = max(1, int((price_df.index[-1] - price_df.index[0]).days) + 1)

    if strategy_names is None:
        strategy_map = BENCHMARK_STRATEGY_MAP.copy()
    else:
        strategy_map = {
            name: STRATEGY_MAP[name]
            for name in strategy_names
            if name in STRATEGY_MAP
        }

    for ai_spec in ai_models or []:
        name = ai_spec.get("name")
        path = ai_spec.get("path")
        threshold = float(ai_spec.get("threshold", 0.001))
        if name and path:
            strategy_map[name] = _make_ai_strategy(name, path, threshold=threshold)

    def build_entry_exit_signals(signal, index):
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

    for name, strat in strategy_map.items():
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
        pf = vbt.Portfolio.from_signals(
            price_df,
            entries,
            exits,
            init_cash=cash,
            freq="1D",
        )

        stats = pf.stats(metrics=["total_return", "win_rate", "max_dd", "sharpe_ratio"], agg_func=None)
        if isinstance(stats, pd.DataFrame) and "Total Return [%]" not in stats.index:
            stats = stats.T

        def build_annualized_returns(total_return_pct, period_days=None):
            total_return = _coerce_numeric_series(total_return_pct) / 100.0
            if period_days is None:
                period_days = observation_period_days
            period_days = _coerce_numeric_series(period_days)
            if hasattr(period_days, "replace"):
                period_days = period_days.replace(0, np.nan)
            elif period_days == 0:
                period_days = np.nan
            annual_factor = 365.0 / period_days
            annual_return = ((1 + total_return) ** annual_factor - 1) * 100
            if hasattr(annual_return, "fillna"):
                return annual_return.fillna(np.nan)
            return annual_return

        try:
            annualized_returns = build_annualized_returns(
                stats.loc["Total Return [%]"], observation_period_days
            )
            if isinstance(annualized_returns, pd.Series):
                annualized_returns = annualized_returns.replace([np.inf, -np.inf], np.nan)
            stats.loc["Annualized Return [%]"] = annualized_returns
            stats.loc["CAGR [%]"] = annualized_returns
        except Exception as exc:
            raise ValueError(f"Failed to compute annualized returns for strategy '{name}': {exc}") from exc

        summary_metrics = [
            "Total Return [%]",
            "Annualized Return [%]",
            "CAGR [%]",
            "Max Drawdown [%]",
            "Win Rate [%]",
            "Sharpe Ratio",
        ]

        selected_stats = stats.loc[summary_metrics]
        pf_summary = selected_stats.mean(axis=1) if isinstance(selected_stats, pd.DataFrame) else selected_stats

        summary[name] = pf_summary
        per_symbol[name] = {"summary": pf_summary, "stats": stats}
        portfolios[name] = pf

    result_summary = pd.DataFrame(summary).T
    return portfolios, result_summary, per_symbol, price_df, entries, exits
