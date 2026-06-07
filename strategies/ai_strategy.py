import logging
import pandas as pd
import numpy as np
from ai.predict import predict_signals_for_series, load_pipeline_model, get_selected_features
from ai.data import features_for_series
from strategies.base import Strategy


class AIStrategy(Strategy):
    name = "ai_model"

    def generate_signals(self, df: pd.DataFrame, model_path: str = "ai/models/lightgbm_model.pkl", threshold: float = 0.001) -> pd.Series:
        """Generate BUY/SELL/HOLD signals for a single-symbol DataFrame expected to have a 'close' column."""
        series = df["close"].astype(float).copy()
        try:
            return predict_signals_for_series(series, model_path=model_path, threshold=threshold)
        except Exception as e:
            logging.warning("AI strategy prediction failed: %s. Falling back to HOLD signals.", e)
            return pd.Series("HOLD", index=series.index)

    def generate_signal(self, df: pd.DataFrame, model_path: str = "ai/models/lightgbm_model.pkl", threshold: float = 0.001) -> str:
        """Generate a single signal for the latest candle using the AI model."""
        sigs = self.generate_signals(df, model_path=model_path, threshold=threshold)
        return sigs.dropna().iloc[-1]

    def generate_signals_pipeline(
        self,
        df: pd.DataFrame,
        model_path: str = "ai/models/pipeline_model.pkl",
        threshold: float = 0.001,
    ) -> pd.Series:
        """Generate signals using the new pipeline model with proper feature selection.
        
        Args:
            df: DataFrame with OHLCV data (must have 'close' column)
            model_path: path to pipeline model
            threshold: return threshold for BUY/SELL signals
        
        Returns:
            Series of BUY/SELL/HOLD signals
        """
        try:
            # Build features
            if "close" not in df.columns:
                raise ValueError("DataFrame must contain 'close' column")
            
            X = features_for_series(df)
            if X.empty:
                logging.warning("No features generated from input data")
                return pd.Series("HOLD", index=df.index)
            
            # Load pipeline and make predictions
            pipeline, metadata = load_pipeline_model(model_path)
            selected_features = metadata.get("selected_features", X.columns.tolist())
            
            # Align features
            X_selected = X[selected_features] if all(f in X.columns for f in selected_features) else X
            
            # Get predictions
            preds = pipeline.predict(X_selected)
            
            # Generate signals
            signals = pd.Series("HOLD", index=df.index)
            signals.loc[X_selected.index] = np.where(
                preds > threshold,
                "BUY",
                np.where(preds < -threshold, "SELL", "HOLD")
            )
            
            return signals
            
        except Exception as e:
            logging.warning("Pipeline strategy prediction failed: %s. Falling back to HOLD signals.", e)
            return pd.Series("HOLD", index=df.index)

    def generate_signal_pipeline(
        self,
        df: pd.DataFrame,
        model_path: str = "ai/models/pipeline_model.pkl",
        threshold: float = 0.001,
    ) -> str:
        """Generate a single signal for the latest candle using pipeline model."""
        sigs = self.generate_signals_pipeline(df, model_path=model_path, threshold=threshold)
        return sigs.dropna().iloc[-1]


strategy = AIStrategy()

