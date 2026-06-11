import logging
import pandas as pd
from ai.predict import _is_pipeline_model, predict_signals_for_model_path, predict_signals_for_series
from strategies.base import Strategy


class AIStrategy(Strategy):
    name = "ai_model"

    def generate_signals(
        self,
        df: pd.DataFrame,
        model_path: str = "ai/models/lightgbm_model.pkl",
        threshold: float = 0.001,
    ) -> pd.Series:
        """Generate BUY/SELL/HOLD signals for a single-symbol DataFrame expected to have a 'close' column."""
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        series = df["close"].astype(float).copy()
        try:
            if model_path and _is_pipeline_model(model_path):
                return predict_signals_for_model_path(series, model_path=model_path, threshold=threshold)
            return predict_signals_for_series(series, model_path=model_path, threshold=threshold)
        except Exception as e:
            logging.warning("AI strategy prediction failed: %s. Falling back to HOLD signals.", e)
            return pd.Series("HOLD", index=series.index)

    def generate_signal(
        self,
        df: pd.DataFrame,
        model_path: str = "ai/models/lightgbm_model.pkl",
        threshold: float = 0.001,
    ) -> str:
        """Generate a single signal for the latest candle using the AI model."""
        sigs = self.generate_signals(df, model_path=model_path, threshold=threshold)
        return sigs.dropna().iloc[-1]


strategy = AIStrategy()

