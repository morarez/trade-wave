# tests/test_strategy.py
import pandas as pd
import numpy as np
import pytest

from strategies.sma_rsi import strategy as sma_rsi_strategy
from strategies.bollinger_rsi import strategy as bollinger_rsi_strategy
from strategies.macd_trend import strategy as macd_trend_strategy
from strategies.ai_strategy import strategy as ai_strategy
from strategies.strategy_factory import get_strategy

ALLOWED_SIGNALS = {"BUY", "SELL", "HOLD"}

@pytest.fixture
def price_data():
    """Generate a mock price series with a close column."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    prices = 100 + np.cumsum(np.random.randn(len(dates)))
    return pd.DataFrame({"close": prices}, index=dates)


@pytest.mark.parametrize(
    "strategy",
    [sma_rsi_strategy, bollinger_rsi_strategy, macd_trend_strategy],
)
def test_strategy_generate_signals(strategy, price_data):
    signals = strategy.generate_signals(price_data)

    assert isinstance(signals, pd.Series)
    assert signals.index.equals(price_data.index)
    assert set(signals.dropna().unique()).issubset(ALLOWED_SIGNALS)
    assert strategy.generate_signal(price_data) == signals.iloc[-1]


def test_ai_strategy_generate_signals(monkeypatch, price_data):
    monkeypatch.setattr(
        "strategies.ai_strategy.predict_signals_for_series",
        lambda series, model_path=None, threshold=None: pd.Series("BUY", index=series.index),
    )

    signals = ai_strategy.generate_signals(price_data)
    assert isinstance(signals, pd.Series)
    assert set(signals.unique()) == {"BUY"}
    assert ai_strategy.generate_signal(price_data) == "BUY"


def test_strategies_handle_missing_data(price_data):
    data_with_nans = price_data.copy()
    data_with_nans.iloc[:10] = np.nan

    for strategy in [sma_rsi_strategy, bollinger_rsi_strategy, macd_trend_strategy, ai_strategy]:
        signals = strategy.generate_signals(data_with_nans)
        assert isinstance(signals, pd.Series)
        assert signals.index.equals(data_with_nans.index)
        assert signals.iloc[-1] in ALLOWED_SIGNALS


def test_get_strategy_by_name():
    assert get_strategy("sma_rsi") is sma_rsi_strategy
    assert get_strategy("bollinger_rsi") is bollinger_rsi_strategy
    assert get_strategy("macd_trend") is macd_trend_strategy
    assert get_strategy("ai_model") is ai_strategy
    assert get_strategy("unknown") is None
