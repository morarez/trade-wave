import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_CONFIG: Dict[str, Any] = {
    "train": {
        "mode": "final",
        "symbols": ["AAPL", "MSFT", "GOOG"],
        "start": "2015-01-01",
        "model_path": "ai/models/pipeline_model.pkl",
        "train_size": 500,
        "test_size": 100,
        "step_size": 50,
        "verbose": False,
    },
    "backtest": {
        "symbols": "AAPL,MSFT,GOOG",
        "start": "2024-01-01",
        "end": None,
        "cash": 10000,
        "interval": "1d",
        "strategies": "",
    },
    "api": {
        "host": "127.0.0.1",
        "port": 5000,
        "debug": True,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Optional[Path | str] = None) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)

    if config_path is None:
        config_path = Path("config.json")
    if not config_path:
        return config

    path = Path(config_path)
    if not path.exists():
        return config

    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    if not isinstance(loaded, dict):
        raise ValueError("Configuration file must contain a JSON object")

    return _deep_merge(config, loaded)
