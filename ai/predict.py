from typing import Optional
import numpy as np
import pandas as pd

from .model import load_model
from .data import features_for_series


def predict_signals_for_series(series: pd.Series, model_path: str = "ai/models/lightgbm_model.pkl", threshold: float = 0.001):
    """Return a pd.Series of signals ('BUY','SELL','HOLD') for the given close price series.

    threshold: minimum predicted return to take a long/short signal.
    """
    model = load_model(model_path)
    feats = features_for_series(series)
    if feats.empty:
        return pd.Series("HOLD", index=series.index)

    X = feats.select_dtypes(["number"]).fillna(0)
    preds = model.predict(X)
    preds_ser = pd.Series(preds, index=X.index)

    signals = pd.Series("HOLD", index=series.index)
    # Align predictions back onto the full index
    signals.loc[preds_ser.index] = np.where(preds_ser > threshold, "BUY", np.where(preds_ser < -threshold, "SELL", "HOLD"))
    return signals
