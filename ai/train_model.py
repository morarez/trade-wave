"""Simple training script to build a LightGBM model using historical data.

Usage:
    python ai/train_model.py

This script will:
- fetch all symbols listed in backtest.run_all_backtests default
- build features and targets
- train a LightGBM regressor and save to `ai/models/lightgbm_model.pkl`
"""
import logging
import os
import sys

# Ensure project root is on sys.path so imports work when running this file as a script
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data_handler import get_yfinance_data
from ai.data import build_dataset
from ai.model import train_model


def main():
    """Train a LightGBM model on historical price data.
    
    Fetches AAPL, MSFT, GOOG data from 2015 to today, builds features and targets,
    trains a model, and saves it to ai/models/lightgbm_model.pkl.
    
    Logs the model path and test MSE upon completion.
    """
    logging.basicConfig(level=logging.INFO)


    symbols = ["AAPL", "MSFT", "GOOG"]
    price_df = get_yfinance_data(symbols=symbols, start="2015-01-01")
    X, y = build_dataset(price_df)
    model_path, mse = train_model(X, y)
    logging.info(f"Trained model saved to {model_path}; test_mse={mse:.6f}")


if __name__ == "__main__":
    main()
