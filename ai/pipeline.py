"""Training pipeline with walk-forward validation, feature selection, and scaling.

This module provides a complete ML pipeline that handles:
- Walk-forward/rolling validation windows for time-series data
- Feature selection using mutual information and variance thresholds
- Feature scaling (StandardScaler)
- Model training with comprehensive metadata tracking
- Metadata persistence (feature list, scaler, feature selection info)
"""

import json
import os
from typing import Tuple, Dict, List, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression, VarianceThreshold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class TimeSeriesPipeline:
    """Complete ML training pipeline for time-series forecasting with walk-forward validation."""

    def __init__(
        self,
        models_dir: str = "ai/models",
        n_features: int = 20,
        variance_threshold: float = 0.01,
        verbose: bool = False,
    ):
        """Initialize the pipeline.

        Args:
            models_dir: directory to save models and metadata.
            n_features: number of features to select via SelectKBest.
            variance_threshold: minimum variance threshold for features.
            verbose: whether to log progress.
        """
        self.models_dir = models_dir
        self.n_features = n_features
        self.variance_threshold = variance_threshold
        self.verbose = verbose

        os.makedirs(models_dir, exist_ok=True)

        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.metadata = {}

    def _log(self, msg: str):
        if self.verbose:
            print(f"[Pipeline] {msg}")

    def select_features(
        self, X: pd.DataFrame, y: pd.Series, method: str = "mutual_info"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select features using variance threshold + SelectKBest.

        Args:
            X: feature DataFrame.
            y: target Series.
            method: "mutual_info" for mutual information, "tree" for tree-based (not implemented here).

        Returns:
            Tuple of (X_selected, selected_feature_names).
        """
        self._log(f"Selecting top {self.n_features} features from {X.shape[1]} candidates...")

        # Keep only numeric features and ignore non-numeric metadata columns.
        X = X.select_dtypes(include=[np.number]).copy()
        if X.empty:
            raise ValueError("No numeric features found for feature selection.")

        # Step 1: Remove low-variance features
        variance_filter = VarianceThreshold(threshold=self.variance_threshold)
        X_var = variance_filter.fit_transform(X)
        var_feature_names = X.columns[variance_filter.get_support()].tolist()
        self._log(f"After variance filtering: {len(var_feature_names)} features")

        # Step 2: SelectKBest with mutual information
        selector = SelectKBest(
            score_func=mutual_info_regression, k=min(self.n_features, len(var_feature_names))
        )
        X_selected = selector.fit_transform(
            pd.DataFrame(X_var, columns=var_feature_names, index=X.index), y
        )
        selected_feature_names = [
            var_feature_names[i] for i in selector.get_support(indices=True)
        ]

        self._log(f"Selected features: {selected_feature_names}")
        self.feature_selector = (variance_filter, selector)

        return pd.DataFrame(X_selected, columns=selected_feature_names, index=X.index), selected_feature_names

    def scale_features(
        self, X: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """Scale features using StandardScaler.

        Args:
            X: feature DataFrame.
            fit: if True, fit the scaler; if False, transform only.

        Returns:
            Scaled feature DataFrame.
        """
        if fit:
            self._log("Fitting StandardScaler...")
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_scaled = self.scaler.transform(X)

        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    def walk_forward_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_size: int,
        test_size: int,
        step_size: int = 1,
    ) -> Dict[str, Any]:
        """Perform walk-forward validation on time-series data.

        Args:
            X: feature DataFrame (already sorted by time).
            y: target Series (already sorted by time).
            train_size: number of samples in each training window.
            test_size: number of samples in each testing window.
            step_size: number of samples to move forward each iteration.

        Returns:
            Dictionary with metrics and predictions for each fold.
        """
        n_samples = len(X)
        folds_data = []
        all_test_preds = []
        all_test_true = []

        fold_idx = 0
        start_idx = 0

        self._log(
            f"Starting walk-forward validation: "
            f"train_size={train_size}, test_size={test_size}, step_size={step_size}"
        )

        while start_idx + train_size + test_size <= n_samples:
            fold_idx += 1
            train_end_idx = start_idx + train_size
            test_end_idx = train_end_idx + test_size

            X_train_fold = X.iloc[start_idx:train_end_idx]
            y_train_fold = y.iloc[start_idx:train_end_idx]
            X_test_fold = X.iloc[train_end_idx:test_end_idx]
            y_test_fold = y.iloc[train_end_idx:test_end_idx]

            # Select and scale features on train fold
            X_train_selected, _ = self.select_features(X_train_fold, y_train_fold)
            X_train_scaled = self.scale_features(X_train_selected, fit=True)

            # Apply same transformations to test fold
            X_test_selected = X_test_fold[X_train_selected.columns]
            X_test_scaled = self.scale_features(X_test_selected, fit=False)

            # Train model on this fold
            model_fold = lgb.LGBMRegressor(verbose=-1)
            model_fold.fit(X_train_scaled, y_train_fold)

            # Predict on test fold
            preds_fold = model_fold.predict(X_test_scaled)

            # Compute metrics
            mse = mean_squared_error(y_test_fold, preds_fold)
            mae = mean_absolute_error(y_test_fold, preds_fold)
            r2 = r2_score(y_test_fold, preds_fold)

            fold_info = {
                "fold": fold_idx,
                "train_idx_range": (start_idx, train_end_idx),
                "test_idx_range": (train_end_idx, test_end_idx),
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "n_train": len(X_train_fold),
                "n_test": len(X_test_fold),
            }
            folds_data.append(fold_info)
            all_test_preds.extend(preds_fold)
            all_test_true.extend(y_test_fold.values)

            self._log(
                f"Fold {fold_idx}: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.4f} "
                f"(train={len(X_train_fold)}, test={len(X_test_fold)})"
            )

            start_idx += step_size

        # Summary metrics across all folds
        overall_mse = mean_squared_error(all_test_true, all_test_preds)
        overall_mae = mean_absolute_error(all_test_true, all_test_preds)
        overall_r2 = r2_score(all_test_true, all_test_preds)

        self._log(
            f"Walk-forward summary: MSE={overall_mse:.6f}, "
            f"MAE={overall_mae:.6f}, R²={overall_r2:.4f} ({fold_idx} folds)"
        )

        return {
            "folds": folds_data,
            "overall_mse": overall_mse,
            "overall_mae": overall_mae,
            "overall_r2": overall_r2,
            "n_folds": fold_idx,
            "predictions": all_test_preds,
            "actuals": all_test_true,
        }

    def train_final_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """Train final model on all data (with train/validation/test split).

        This method:
        1. Selects features based on full dataset
        2. Scales features
        3. Splits into train/validation/test
        4. Trains model on train set
        5. Evaluates on validation and test sets
        6. Returns metrics and model info

        Args:
            X: feature DataFrame.
            y: target Series.
            test_size: fraction for test set; remaining split between train/validation.

        Returns:
            Dictionary with training results and metadata.
        """
        self._log(f"Training final model on {len(X)} samples...")

        # Split data (preserving time order)
        n_samples = len(X)
        test_idx = int(n_samples * (1 - test_size))
        val_idx = int(test_idx * 0.8)  # 80% of non-test for train, 20% for validation

        X_train = X.iloc[:val_idx]
        y_train = y.iloc[:val_idx]
        X_val = X.iloc[val_idx:test_idx]
        y_val = y.iloc[val_idx:test_idx]
        X_test = X.iloc[test_idx:]
        y_test = y.iloc[test_idx:]

        self._log(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

        # Feature selection on training data
        X_train_selected, selected_features = self.select_features(X_train, y_train)

        # Scale features
        X_train_scaled = self.scale_features(X_train_selected, fit=True)

        # Apply transformations to validation and test
        X_val_selected = X_val[selected_features]
        X_val_scaled = self.scale_features(X_val_selected, fit=False)

        X_test_selected = X_test[selected_features]
        X_test_scaled = self.scale_features(X_test_selected, fit=False)

        # Train final model
        self.model = lgb.LGBMRegressor(verbose=-1)
        self.model.fit(X_train_scaled, y_train)

        # Evaluate on all sets
        val_preds = self.model.predict(X_val_scaled)
        test_preds = self.model.predict(X_test_scaled)

        train_mse = mean_squared_error(y_train, self.model.predict(X_train_scaled))
        val_mse = mean_squared_error(y_val, val_preds)
        test_mse = mean_squared_error(y_test, test_preds)

        train_r2 = r2_score(y_train, self.model.predict(X_train_scaled))
        val_r2 = r2_score(y_val, val_preds)
        test_r2 = r2_score(y_test, test_preds)

        metrics = {
            "train_mse": train_mse,
            "val_mse": val_mse,
            "test_mse": test_mse,
            "train_r2": train_r2,
            "val_r2": val_r2,
            "test_r2": test_r2,
        }

        self._log(
            f"Final model - Train MSE: {train_mse:.6f}, Val MSE: {val_mse:.6f}, Test MSE: {test_mse:.6f}"
        )
        self._log(f"Final model - Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}")

        # Store metadata
        self.metadata = {
            "timestamp": datetime.now().isoformat(),
            "selected_features": selected_features,
            "n_selected_features": len(selected_features),
            "feature_importance": dict(
                zip(selected_features, self.model.feature_importances_.tolist())
            ),
            "metrics": metrics,
            "split_sizes": {
                "train": len(X_train),
                "validation": len(X_val),
                "test": len(X_test),
            },
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_std": self.scaler.scale_.tolist(),
        }

        return {
            "model": self.model,
            "metrics": metrics,
            "metadata": self.metadata,
            "selected_features": selected_features,
        }

    def save_model(self, model_path: str = None):
        """Save trained model and metadata.

        Args:
            model_path: path to save model (default: ai/models/pipeline_model.pkl).
        """
        if model_path is None:
            model_path = os.path.join(self.models_dir, "pipeline_model.pkl")

        if self.model is None:
            raise ValueError("No model trained yet. Call train_final_model() first.")

        # Save model artifacts
        artifacts = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_selector": self.feature_selector,
            "metadata": self.metadata,
        }

        joblib.dump(artifacts, model_path)
        self._log(f"Model saved to {model_path}")

        # Save metadata as JSON for inspection
        metadata_path = model_path.replace(".pkl", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        self._log(f"Metadata saved to {metadata_path}")

        return model_path, metadata_path

    def load_model(self, model_path: str = None):
        """Load trained model and metadata.

        Args:
            model_path: path to load model from.

        Returns:
            Tuple of (model, metadata).
        """
        if model_path is None:
            model_path = os.path.join(self.models_dir, "pipeline_model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        artifacts = joblib.load(model_path)
        self.model = artifacts.get("model")
        self.scaler = artifacts.get("scaler")
        self.feature_selector = artifacts.get("feature_selector")
        self.metadata = artifacts.get("metadata", {})

        self._log(f"Model loaded from {model_path}")
        return self.model, self.metadata

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data using trained model.

        Args:
            X: feature DataFrame (must contain same features as training).

        Returns:
            Predictions.
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() or train_final_model() first.")

        selected_features = self.metadata.get("selected_features", X.columns.tolist())
        X_selected = X[selected_features]
        X_scaled = self.scaler.transform(X_selected)

        return self.model.predict(X_scaled)
