import os
from typing import Optional

import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def train_model(X, y, model_path: str = "ai/models/lightgbm_model.pkl", test_size: float = 0.2, random_state: int = 42, **lgb_params):
    """Train a LightGBM regressor on features and targets.
    
    Splits data into train/test (no shuffling to preserve time order), trains the model,
    and saves it to disk using joblib.
    
    Args:
        X: feature DataFrame (may contain non-numeric columns which are dropped).
        y: target Series (next-period returns).
        model_path: where to save the trained model (default ai/models/lightgbm_model.pkl).
        test_size: fraction of data for testing (default 0.2).
        random_state: random seed (default 42).
        **lgb_params: additional LightGBM hyperparameters.
    
    Returns:
        Tuple of (model_path, test_mse).
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Basic preprocessing: drop non-numeric columns (symbol) and keep features
    X_proc = X.select_dtypes(["number"]).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=test_size, shuffle=False
    )

    model = lgb.LGBMRegressor(**lgb_params) if lgb_params else lgb.LGBMRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    # Save both model and the feature columns used during training so
    # prediction can be aligned to the same schema later.
    payload = {"model": model, "feature_names": list(X_proc.columns)}
    joblib.dump(payload, model_path)
    return model_path, mse


def load_model(model_path: str = "ai/models/lightgbm_model.pkl"):
    """Load a trained model from disk.
    
    Args:
        model_path: path to the saved model pickle file.
    
    Returns:
        Trained model object (LGBMRegressor or RandomForestRegressor).
    
    Raises:
        FileNotFoundError: if model file does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    payload = joblib.load(model_path)
    # Backwards compatibility: older files may contain the raw model object.
    if hasattr(payload, "predict"):
        # Try to infer training feature names from the model object for
        # backwards compatibility with older saved models.
        inferred = None
        try:
            # scikit-learn API: number of input features
            if hasattr(payload, "feature_name_"):
                inferred = list(getattr(payload, "feature_name_"))
        except Exception:
            inferred = None
        try:
            # LightGBM booster: feature_name() method
            if inferred is None and hasattr(payload, "booster_"):
                b = getattr(payload, "booster_")
                if hasattr(b, "feature_name"):
                    inferred = list(b.feature_name())
        except Exception:
            inferred = inferred or None
        try:
            # sklearn estimators often expose n_features_in_
            if inferred is None and hasattr(payload, "n_features_in_"):
                n = int(getattr(payload, "n_features_in_"))
                inferred = [f"f{i}" for i in range(n)]
        except Exception:
            inferred = inferred or None
        return payload, inferred
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload.get("feature_names")
    # Unknown format — return as-is (likely will fail later)
    return payload, None
