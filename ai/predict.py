from typing import Optional
import numpy as np
import pandas as pd

from .model import load_model
from .data import features_for_series


def predict_signals_for_series(series: pd.Series, model_path: str = "ai/models/lightgbm_model.pkl", threshold: float = 0.001):
    """Return a pd.Series of signals ('BUY','SELL','HOLD') for the given close price series.

    threshold: minimum predicted return to take a long/short signal.
    """
    model, feature_names = load_model(model_path)
    feats = features_for_series(series)
    if feats.empty:
        return pd.Series("HOLD", index=series.index)
    X_all = feats.select_dtypes(["number"]).fillna(0)

    # If the saved model included the feature names used during training,
    # align the prediction DataFrame to that ordering. Missing features are
    # filled with zeros; extra features are ignored.
    if feature_names is not None:
        # create an aligned DataFrame with exactly the training columns
        X = pd.DataFrame(0, index=X_all.index, columns=feature_names)
        for col in X_all.columns.intersection(feature_names):
            X[col] = X_all[col]
    else:
        # If we don't have explicit feature names, ensure the numeric matrix
        # has the same number of columns the model expects (if known).
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
    # Align predictions back onto the full index
    signals.loc[preds_ser.index] = np.where(preds_ser > threshold, "BUY", np.where(preds_ser < -threshold, "SELL", "HOLD"))
    return signals
