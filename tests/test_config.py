import json

from config import load_config


def test_load_config_uses_builtin_defaults():
    config = load_config(config_path=None)

    assert config["train"]["model_path"] == "ai/models/pipeline_model.pkl"
    assert config["backtest"]["cash"] == 10000
    assert config["api"]["port"] == 5000


def test_load_config_merges_json_overrides(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"backtest": {"cash": 5000}}))

    config = load_config(config_path=config_path)

    assert config["backtest"]["cash"] == 5000
    assert config["train"]["model_path"] == "ai/models/pipeline_model.pkl"
