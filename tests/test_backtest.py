import pandas as pd
import pytest

from backtest import run_all_backtests


@pytest.fixture
def price_data():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02"])
    return pd.DataFrame({"AAPL": [100.0, 101.0]}, index=dates)


class FakeStrategy:
    def generate_signals(self, df):
        return "HOLD"


def test_per_symbol_pf_is_a_summary_dataframe(monkeypatch, price_data):
    monkeypatch.setattr("backtest.get_yfinance_data", lambda *args, **kwargs: price_data)
    monkeypatch.setattr("backtest.STRATEGY_MAP", {"fake": FakeStrategy()})

    portfolios, _, per_symbol, _, _, _ = run_all_backtests(
        symbols=["AAPL"],
        start_date="2020-01-01",
        end_date="2020-01-02",
    )

    assert isinstance(portfolios, dict)
    assert "fake" in portfolios
    assert hasattr(portfolios["fake"], "stats")
    assert "fake" in per_symbol
    assert isinstance(per_symbol["fake"], dict)
    assert "summary" in per_symbol["fake"]
    assert "stats" in per_symbol["fake"]
    assert isinstance(per_symbol["fake"]["summary"], pd.Series)
    # new summary metrics should be present
    for key in [
        "Total Return [%]",
        "Annualized Return [%]",
        "CAGR [%]",
        "Max Drawdown [%]",
        "Win Rate [%]",
    ]:
        assert key in per_symbol["fake"]["summary"].index
    assert isinstance(per_symbol["fake"]["stats"], pd.DataFrame)
