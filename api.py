from flask import Flask, jsonify, request
import pandas as pd
import logging
import tempfile
from datetime import datetime

from backtest import run_all_backtests
from ai.train_model import train_with_final_split, train_with_walk_forward
from ai.predict import predict_signals_for_model_path
from data_handler import get_yfinance_data
from strategies.strategy_factory import STRATEGY_MAP

logging.basicConfig(level=logging.INFO)


def parse_symbol_list(symbol_string):
    if not symbol_string:
        return None
    if isinstance(symbol_string, list):
        symbols = [part.strip().upper() for part in symbol_string if part and str(part).strip()]
    else:
        symbols = [part.strip().upper() for part in str(symbol_string).replace(",", " ").split() if part.strip()]
    return symbols or None


def parse_model_list(models_input):
    if not models_input:
        return []
    models = []
    if isinstance(models_input, str):
        models_input = [models_input]

    for entry in models_input:
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get("path")
            path = entry.get("path")
            threshold = entry.get("threshold", 0.001)
        else:
            entry = str(entry)
            if "=" in entry:
                name, path = [part.strip() for part in entry.split("=", 1)]
            else:
                path = entry.strip()
                name = path.split("/")[-1].split(".")[0]
            threshold = 0.001
        if path:
            models.append({"name": name, "path": path, "threshold": threshold})
    return models


def normalize_list(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def serialize_dataframe(df: pd.DataFrame):
    df = df.copy()
    columns = [str(col) for col in df.columns]
    index = [str(idx) for idx in df.index]
    data = []

    def normalize_value(value):
        if pd.isna(value):
            return None

        if isinstance(value, pd.Timedelta):
            return str(value)
        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        try:
            import numpy as _np
            if isinstance(value, (_np.floating, _np.integer, _np.bool_)):
                value = value.item()
        except Exception:
            pass

        if isinstance(value, (float, int, bool)):
            if isinstance(value, bool):
                return value
            if pd.isna(value):
                return None
            if value == float("inf") or value == float("-inf"):
                raise ValueError(f"Encountered non-finite numeric value: {value}")
            if isinstance(value, float) and (value != value):
                raise ValueError(f"Encountered NaN numeric value: {value}")
            return value

        if isinstance(value, (list, tuple)):
            return [normalize_value(v) for v in value]
        return value

    for row in df.itertuples(index=False, name=None):
        data.append([normalize_value(value) for value in row])
    return {"columns": columns, "index": index, "data": data}


def serialize_results(results):
    return {
        "summary": serialize_dataframe(results["summary"]),
        "per_symbol": {
            name: {
                "summary": serialize_dataframe(data["summary"].to_frame().T),
                "stats": serialize_dataframe(data["stats"]),
            }
            for name, data in results["per_symbol"].items()
        },
    }


def create_app():
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def health_check():
        return jsonify({"status": "ok", "message": "Trade Wave API is running. Use POST /api/backtest to run backtests."})

    @app.route("/api/backtest", methods=["GET", "POST"])
    def api_backtest():
        if request.method == "GET":
            return jsonify({
                "message": "POST JSON data to run backtests",
                "example_request": {
                    "symbols": "AAPL,MSFT,GOOG",
                    "strategies": "sma_rsi,bollinger_rsi",
                    "start": "2025-01-01",
                    "end": "2024-06-01",
                    "cash": 10000,
                    "interval": "1d",
                    "models": ["ai_model=ai/models/pipeline_model.pkl"]
                },
                "notes": {
                    "symbols": "Comma-separated stock symbols (optional, defaults to AAPL)",
                    "strategies": "Comma-separated strategy names (optional, defaults to all)",
                    "start": "Start date YYYY-MM-DD (optional, default: 2025-01-01)",
                    "end": "End date YYYY-MM-DD (optional, default: today)",
                    "cash": "Starting cash (optional, default: 10000)",
                    "interval": "Data interval (optional, default: 1d)",
                    "models": "List of AI model paths in format 'name=path' (optional)"
                }
            })

        payload = request.get_json(silent=True) or {}
        symbols = payload.get("symbols")
        strategy_names = normalize_list(payload.get("strategies"))
        ai_models = parse_model_list(payload.get("models"))
        start_date = payload.get("start", "2025-01-01")
        end_date = payload.get("end")
        cash = payload.get("cash", 10000)
        interval = payload.get("interval", "1d")
        symbol_list = parse_symbol_list(symbols)

        try:
            _, summary, per_symbol, _, _, _ = run_all_backtests(
                symbols=symbol_list,
                strategy_names=strategy_names,
                ai_models=ai_models,
                start_date=start_date,
                end_date=end_date,
                cash=cash,
                interval=interval,
            )
            return jsonify(serialize_results({"summary": summary, "per_symbol": per_symbol}))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/predict", methods=["GET", "POST"])
    def api_predict():
        if request.method == "GET":
            return jsonify({
                "message": "POST JSON data to generate a prediction",
                "example_request": {
                    "symbol": "AAPL",
                    "model_path": "ai/models/pipeline_model.pkl",
                    "strategies": "sma_rsi,bollinger_rsi"
                }
            })

        payload = request.get_json(silent=True) or {}
        symbol = (payload.get("symbol") or "AAPL").strip().upper()
        model_path = payload.get("model_path", "ai/models/pipeline_model.pkl")
        strategies = normalize_list(payload.get("strategies")) or []

        try:
            if not symbol:
                return jsonify({"error": "symbol is required"}), 400

            price_df = get_yfinance_data(symbols=[symbol], start="2025-01-01")
            if price_df.empty or symbol not in price_df.columns:
                return jsonify({"error": f"No price data available for {symbol}"}), 404

            series = price_df[symbol].dropna()
            if series.empty:
                return jsonify({"error": f"No price data available for {symbol}"}), 404

            ai_signal = predict_signals_for_model_path(series, model_path=model_path, threshold=0.001)
            ai_signal = ai_signal.dropna()
            latest_ai_signal = ai_signal.iloc[-1] if not ai_signal.empty else "HOLD"

            strategy_summary = []
            for strategy_name in strategies:
                if strategy_name in STRATEGY_MAP:
                    strategy_summary.append(f"{strategy_name}: {latest_ai_signal}")
                else:
                    strategy_summary.append(f"{strategy_name}: unknown")

            latest_price = float(series.iloc[-1])
            previous_price = float(series.iloc[-2]) if len(series) > 1 else latest_price
            change_pct = ((latest_price - previous_price) / previous_price * 100) if previous_price else 0.0

            strategy_signals = [
                {"name": strategy_name, "signal": latest_ai_signal if strategy_name in STRATEGY_MAP else "unknown"}
                for strategy_name in strategies
            ]

            return jsonify({
                "status": "success",
                "message": (
                    f"{symbol} latest close: ${latest_price:.2f} "
                    f"(change {change_pct:+.2f}%). AI signal: {latest_ai_signal}."
                ),
                "symbol": symbol,
                "latest_price": latest_price,
                "change_pct": change_pct,
                "ai_signal": latest_ai_signal,
                "strategies": strategies,
                "strategy_signals": strategy_signals,
            })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/train", methods=["GET", "POST"])
    def api_train():
        if request.method == "GET":
            return jsonify({
                "message": "POST JSON data to train an AI model",
                "example_request": {
                    "mode": "final",
                    "symbols": "AAPL,MSFT,GOOG",
                    "start": "2015-01-01",
                    "model_path": "ai/models/pipeline_model.pkl"
                },
                "modes": {
                    "final": "Train on full data with final split",
                    "walk_forward": "Train with walk-forward validation"
                },
                "walk_forward_params": {
                    "train_size": "Training window size (default: 500)",
                    "test_size": "Test window size (default: 100)",
                    "step_size": "Step size for walk-forward (default: 50)"
                }
            })

        payload = request.get_json(silent=True) or {}
        mode = payload.get("mode", "final")
        symbols = payload.get("symbols", "AAPL,MSFT,GOOG")
        start = payload.get("start", "2015-01-01")
        model_path = payload.get("model_path", "ai/models/pipeline_model.pkl")
        
        try:
            symbol_list = parse_symbol_list(symbols)
            if mode == "final":
                train_with_final_split(
                    symbols=symbol_list,
                    start=start,
                    model_path=model_path,
                    verbose=False,
                )
                return jsonify({"status": "success", "message": f"Model trained and saved to {model_path}"})
            elif mode == "walk_forward":
                train_size = payload.get("train_size", 500)
                test_size = payload.get("test_size", 100)
                step_size = payload.get("step_size", 50)
                train_with_walk_forward(
                    symbols=symbol_list,
                    start=start,
                    train_size=train_size,
                    test_size=test_size,
                    step_size=step_size,
                    model_path=model_path,
                    verbose=False,
                )
                return jsonify({"status": "success", "message": f"Model trained with walk-forward validation and saved to {model_path}"})
            else:
                return jsonify({"error": f"Unknown mode: {mode}"}), 400
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app


def run_server(host: str = "127.0.0.1", port: int = 5000, debug: bool = True):
    app = create_app()
    app.run(host=host, port=port, debug=debug)
