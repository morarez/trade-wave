from typing import Any, Dict, List
import os
import joblib
import numpy as np
import pandas as pd

from .model import load_model
from .data import features_for_series


def load_pipeline_model(
    model_path: str = "ai/models/pipeline_model.pkl",
) -> tuple:
    """Load a trained pipeline model and its metadata.

    Args:
        model_path: path to the saved pipeline model.

    Returns:
        Tuple of (pipeline, metadata).
    """
    from .pipeline import TimeSeriesPipeline

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    pipeline = TimeSeriesPipeline(verbose=False)
    model, metadata = pipeline.load_model(model_path)

    return pipeline, metadata


def get_selected_features(
    model_path: str = "ai/models/pipeline_model.pkl",
) -> List[str]:
    _, metadata = load_pipeline_model(model_path)
    return metadata.get("selected_features", [])


def get_model_metrics(
    model_path: str = "ai/models/pipeline_model.pkl",
) -> Dict[str, Any]:
    _, metadata = load_pipeline_model(model_path)
    return metadata.get("metrics", {})


def _is_pipeline_model(model_path: str) -> bool:
    if not os.path.exists(model_path):
        return False
    try:
        payload = joblib.load(model_path)
    except Exception:
        return False
    return isinstance(payload, dict) and "feature_selector" in payload and "metadata" in payload


def _predict_signals_with_pipeline(series: pd.Series, model_path: str, threshold: float = 0.001):
    pipeline, metadata = load_pipeline_model(model_path)
    feats = features_for_series(series)
    if feats.empty:
        return pd.Series("HOLD", index=series.index)

    selected_features = metadata.get("selected_features", list(feats.columns))
    X_selected = feats.reindex(columns=selected_features, fill_value=0)
    preds = pipeline.predict(X_selected)
    preds_ser = pd.Series(preds, index=X_selected.index)

    signals = pd.Series("HOLD", index=series.index)
    signals.loc[preds_ser.index] = np.where(preds_ser > threshold, "BUY", np.where(preds_ser < -threshold, "SELL", "HOLD"))
    return signals


def predict_signals_for_series(
    series: pd.Series,
    model_path: str = "ai/models/lightgbm_model.pkl",
    threshold: float = 0.001,
):
    model, feature_names = load_model(model_path)
    feats = features_for_series(series)
    if feats.empty:
        return pd.Series("HOLD", index=series.index)
    X_all = feats.select_dtypes(["number"]).fillna(0)

    if feature_names is not None:
        X = pd.DataFrame(0, index=X_all.index, columns=feature_names)
        for col in X_all.columns.intersection(feature_names):
            X[col] = X_all[col]
    else:
        if hasattr(model, "n_features_in_"):
            n_in = int(getattr(model, "n_features_in_"))
            if X_all.shape[1] != n_in:
                raise ValueError(
                    f"Model expects {n_in} features but input has {X_all.shape[1]}; retrain model or save feature names."
                )
        X = X_all

    preds = model.predict(X)
    preds_ser = pd.Series(preds, index=X.index)

    signals = pd.Series("HOLD", index=series.index)
    signals.loc[preds_ser.index] = np.where(preds_ser > threshold, "BUY", np.where(preds_ser < -threshold, "SELL", "HOLD"))
    return signals


def predict_signals_for_model_path(
    series: pd.Series,
    model_path: str = "ai/models/lightgbm_model.pkl",
    threshold: float = 0.001,
):
    if _is_pipeline_model(model_path):
        return _predict_signals_with_pipeline(series, model_path, threshold)
    return predict_signals_for_series(series, model_path=model_path, threshold=threshold)
