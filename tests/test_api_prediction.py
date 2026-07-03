import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import create_app


def test_predict_endpoint_returns_success(monkeypatch):
    app = create_app()
    client = app.test_client()

    fake_series = pd.Series([100.0, 101.0], index=pd.date_range("2025-01-01", periods=2, freq="D"))

    monkeypatch.setattr("api.get_yfinance_data", lambda symbols, start=None, end=None, interval="1d": pd.DataFrame({symbols[0]: fake_series}))
    monkeypatch.setattr("api.predict_signals_for_model_path", lambda series, model_path="", threshold=0.001: pd.Series(["BUY", "SELL"], index=series.index))

    response = client.post(
        "/api/predict",
        json={"symbol": "AAPL", "model_path": "ai/models/pipeline_model.pkl", "strategies": "sma_rsi,bollinger_rsi"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "success"
    assert payload["symbol"] == "AAPL"
    assert payload["ai_signal"] == "SELL"
    assert payload["strategy_signals"][0]["signal"] == "SELL"
