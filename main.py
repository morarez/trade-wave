from flask import Flask, jsonify, request
import pandas as pd

from backtest import run_all_backtests

app = Flask(__name__)


def parse_symbol_list(symbol_string):
    """Parse a comma/space-separated symbol string into a list."""
    if not symbol_string:
        return None
    symbols = [part.strip().upper() for part in symbol_string.replace(",", " ").split() if part.strip()]
    return symbols or None


def serialize_dataframe(df: pd.DataFrame):
    df = df.copy()
    columns = [str(col) for col in df.columns]
    index = [str(idx) for idx in df.index]
    data = []
    def normalize_value(value):
        if pd.isna(value):
            return None
        # pandas Timedelta -> ISO-ish string
        if isinstance(value, pd.Timedelta):
            return str(value)
        # pandas Timestamp -> ISO string
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        # numpy scalar -> native python
        try:
            import numpy as _np

            if isinstance(value, (_np.floating, _np.integer, _np.bool_)):
                return value.item()
        except Exception:
            pass
        # fallback to string for other non-serializable objects
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


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    payload = request.get_json(silent=True) or {}
    symbols = payload.get("symbols")
    symbol_list = parse_symbol_list(symbols)

    try:
        portfolios, summary, per_symbol, _, _, _ = run_all_backtests(symbols=symbol_list)
        payload = {"summary": summary, "per_symbol": per_symbol}
        return jsonify(serialize_results(payload))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
