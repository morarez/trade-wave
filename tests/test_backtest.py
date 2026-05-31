import pandas as pd

from backtest import run_all_backtests


class FakeStrategy:
    def generate_signals(self, df):
        return "HOLD"


def test_per_symbol_pf_is_a_summary_dataframe(monkeypatch):
    price_df = pd.DataFrame(
        {"AAPL": [100.0, 101.0]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
    )

    monkeypatch.setattr("backtest.get_yfinance_data", lambda *args, **kwargs: price_df)
    monkeypatch.setattr("backtest.STRATEGY_MAP", {"fake": FakeStrategy()})

    _, _, per_symbol, _, _, _ = run_all_backtests(
        symbols=["AAPL"],
        start_date="2020-01-01",
        end_date="2020-01-02",
    )

    assert "fake" in per_symbol
    assert isinstance(per_symbol["fake"]["pf"], (pd.Series, pd.DataFrame))
    assert "Total Return [%]" in per_symbol["fake"]["pf"].index
