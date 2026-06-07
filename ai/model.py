import os
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import json


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


def train_model_with_scaling(
    X,
    y,
    model_path: str = "ai/models/lightgbm_scaled.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
    **lgb_params
) -> Tuple[str, float, Dict[str, Any]]:
    """Train a LightGBM regressor with feature scaling.
    
    Splits data into train/test, applies StandardScaler to features,
    trains the model, and saves model + scaler with metadata.
    
    Args:
        X: feature DataFrame (numeric columns only).
        y: target Series.
        model_path: where to save the trained model.
        test_size: fraction for testing.
        random_state: random seed.
        **lgb_params: additional LightGBM hyperparameters.
    
    Returns:
        Tuple of (model_path, test_mse, metadata).
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Prepare features
    X_proc = X.select_dtypes(["number"]).fillna(0)
    feature_names = list(X_proc.columns)
    
    # Split data (no shuffle for time series)
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=test_size, shuffle=False, random_state=random_state
    )
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = lgb.LGBMRegressor(**lgb_params) if lgb_params else lgb.LGBMRegressor()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    preds = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, preds)
    
    # Save artifacts
    artifacts = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "n_features": len(feature_names),
    }
    joblib.dump(artifacts, model_path)
    
    # Save metadata as JSON
    metadata = {
        "model_path": model_path,
        "test_mse": float(mse),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    
    metadata_path = model_path.replace(".pkl", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return model_path, mse, metadata


def load_model(model_path: str = "ai/models/lightgbm_model.pkl"):
    """Load a trained model from disk.
    
    Handles both old format (raw model) and new format (model + scaler + metadata).
    
    Args:
        model_path: path to the saved model pickle file.
    
    Returns:
        Tuple of (model, feature_names, scaler_or_none).
        - model: trained LGBMRegressor
        - feature_names: list of feature column names
        - scaler: StandardScaler instance if available, else None
    
    Raises:
        FileNotFoundError: if model file does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    payload = joblib.load(model_path)
    
    # New format: dictionary with model, scaler, feature_names, etc.
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        feature_names = payload.get("feature_names", [])
        scaler = payload.get("scaler", None)
        return model, feature_names, scaler
    
    # Old format: raw model object or simple dict
    if hasattr(payload, "predict"):
        # Try to infer training feature names from the model object
        inferred = None
        try:
            if hasattr(payload, "feature_name_"):
                inferred = list(getattr(payload, "feature_name_"))
        except Exception:
            pass
        try:
            if inferred is None and hasattr(payload, "booster_"):
                b = getattr(payload, "booster_")
                if hasattr(b, "feature_name"):
                    inferred = list(b.feature_name())
        except Exception:
            pass
        try:
            if inferred is None and hasattr(payload, "n_features_in_"):
                n = int(getattr(payload, "n_features_in_"))
                inferred = [f"f{i}" for i in range(n)]
        except Exception:
            pass
        return payload, inferred, None
    
    # Fallback for other dict formats
    if isinstance(payload, dict):
        if "feature_names" in payload:
            return payload.get("model"), payload.get("feature_names"), None
    
    # Unknown format — return as-is
    return payload, None, None


def predict_scaled(
    X: pd.DataFrame,
    model_path: str = "ai/models/lightgbm_scaled.pkl",
) -> pd.Series:
    """Make predictions using a scaled model.
    
    Args:
        X: feature DataFrame (should match training features).
        model_path: path to the saved model.
    
    Returns:
        Series of predictions.
    """
    model, feature_names, scaler = load_model(model_path)
    
    if scaler is None:
        raise ValueError(f"Model at {model_path} does not have a scaler. Use load_model() instead.")
    
    # Select only the features used during training
    if feature_names:
        X_selected = X[feature_names]
    else:
        X_selected = X.select_dtypes(["number"])
    
    # Scale and predict
    X_scaled = scaler.transform(X_selected)
    preds = model.predict(X_scaled)
    
    return pd.Series(preds, index=X.index)
