# tests/test_strategy.py
import pandas as pd
import numpy as np
import pytest
from indicators import compute_indicators, generate_signals_from_df, compute_indicators_vbt, build_signals_vbt
from strategy import generate_signals

@pytest.fixture
def sample_data():
    """Generate a mock dataset for one ticker."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    prices = 100 + np.cumsum(np.random.randn(len(dates)))  # random walk
    data = pd.DataFrame({"TEST": prices}, index=dates)
    return data


def test_indicator_consistency(sample_data):
    """Ensure pandas-ta and vectorbt produce similar indicators."""
    # Compute via pandas-ta
    ind_live = compute_indicators(sample_data)
    df_live = ind_live["TEST"]

    # Compute via vectorbt
    fast, slow, rsi = compute_indicators_vbt(sample_data)

    # Compare last values
    live_last = df_live.iloc[-1]
    vbt_last = {
        "sma_short": fast["TEST"].iloc[-1],
        "sma_long": slow["TEST"].iloc[-1],
        "rsi": rsi["TEST"].iloc[-1],
    }

    assert np.isclose(live_last["sma_short"], vbt_last["sma_short"], rtol=1e-3, equal_nan=True)
    assert np.isclose(live_last["sma_long"], vbt_last["sma_long"], rtol=1e-3, equal_nan=True)
    assert np.isclose(live_last["rsi"], vbt_last["rsi"], rtol=1e-3, equal_nan=True)


def test_signal_consistency(sample_data):
    """Verify that live and vectorbt signals agree on buy/sell at the last candle."""
    # Live signals
    live_signals = generate_signals(sample_data)
    live_signal = live_signals["TEST"]["signal"]

    # Vectorbt signals
    fast, slow, rsi = compute_indicators_vbt(sample_data)
    entries, exits = build_signals_vbt(sample_data, fast, slow, rsi)

    # Determine equivalent last signal
    if entries["TEST"].iloc[-1]:
        vbt_signal = "BUY"
    elif exits["TEST"].iloc[-1]:
        vbt_signal = "SELL"
    else:
        vbt_signal = "HOLD"

    assert live_signal == vbt_signal, f"Live={live_signal}, VBT={vbt_signal}"


def test_nan_handling(sample_data):
    """Ensure indicator NaNs don't break signal generation."""
    data_with_nans = sample_data.copy()
    data_with_nans.iloc[0:10] = np.nan
    signals = generate_signals(data_with_nans)
    assert "TEST" in signals
    assert signals["TEST"]["signal"] in ("BUY", "SELL", "HOLD")
