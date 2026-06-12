"""Training scripts using the ML pipeline with walk-forward validation.

Usage:
    # Train with the basic pipeline (train/validation/test split):
    python ai/train_model.py --mode final
    
    # Or use walk-forward validation:
    python ai/train_model.py --mode walk_forward

This scripts will:
- fetch all symbols listed in backtest.run_all_backtests default
- build features and targets
- apply feature selection and scaling
- train a LightGBM model
- save model, scaler, and metadata
"""
import logging
import os
import sys
import argparse

# Ensure project root is on sys.path so imports work when running this file as a script
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data_handler import get_yfinance_data
from ai.data import build_dataset
from ai.pipeline import TimeSeriesPipeline


def train_with_final_split(
    symbols=None,
    start="2015-01-01",
    model_path="ai/models/pipeline_model.pkl",
    verbose=False,
):
    """Train model using standard train/validation/test split.

    Args:
        symbols: list of stock symbols (default: AAPL, MSFT, GOOG).
        start: start date for historical data.
        model_path: where to save the trained pipeline model.
        verbose: whether to print pipeline logs.

    Returns:
        Path to saved model and metadata.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOG"]

    logger.info(f"Fetching data for {symbols} from {start}...")
    price_df = get_yfinance_data(symbols=symbols, start=start)

    logger.info("Building features and targets...")
    X, y = build_dataset(price_df)
    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")

    pipeline = TimeSeriesPipeline(
        models_dir="ai/models",
        n_features=20,
        variance_threshold=0.01,
        verbose=verbose,
    )

    logger.info("Training final model with train/validation/test split...")
    results = pipeline.train_final_model(X, y, test_size=0.2)

    model_path, metadata_path = pipeline.save_model(model_path)

    logger.info(f"✓ Model saved to {model_path}")
    logger.info(f"✓ Metadata saved to {metadata_path}")
    logger.info(f"✓ Test MSE: {results['metrics']['test_mse']:.6f}")
    logger.info(f"✓ Test R²: {results['metrics']['test_r2']:.4f}")
    logger.info(f"✓ Selected {len(results['selected_features'])} features")

    return model_path, metadata_path


def train_with_walk_forward(
    symbols=None,
    start="2015-01-01",
    train_size=500,
    test_size=100,
    step_size=50,
    model_path="ai/models/pipeline_model.pkl",
    verbose=False,
):
    """Train model using walk-forward validation.

    Args:
        symbols: list of stock symbols.
        start: start date for historical data.
        train_size: number of samples per training window.
        test_size: number of samples per test window.
        step_size: step size for rolling window.
        model_path: where to save the trained pipeline model.
        verbose: whether to print pipeline logs.

    Returns:
        Path to saved model and metadata.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOG"]

    logger.info(f"Fetching data for {symbols} from {start}...")
    price_df = get_yfinance_data(symbols=symbols, start=start)

    logger.info("Building features and targets...")
    X, y = build_dataset(price_df)
    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")

    pipeline = TimeSeriesPipeline(
        models_dir="ai/models",
        n_features=20,
        variance_threshold=0.01,
        verbose=verbose,
    )

    logger.info(
        f"Running walk-forward validation: "
        f"train_size={train_size}, test_size={test_size}, step_size={step_size}..."
    )
    wf_results = pipeline.walk_forward_validation(
        X, y, train_size=train_size, test_size=test_size, step_size=step_size
    )

    logger.info("✓ Walk-forward validation complete:")
    logger.info(f"  - Overall MSE: {wf_results['overall_mse']:.6f}")
    logger.info(f"  - Overall R²: {wf_results['overall_r2']:.4f}")
    logger.info(f"  - Number of folds: {wf_results['n_folds']}")

    logger.info("Training final model on all data...")
    results = pipeline.train_final_model(X, y, test_size=0.2)

    model_path, metadata_path = pipeline.save_model(model_path)

    logger.info(f"✓ Model saved to {model_path}")
    logger.info(f"✓ Metadata saved to {metadata_path}")
    logger.info(f"✓ Final model test MSE: {results['metrics']['test_mse']:.6f}")
    logger.info(f"✓ Final model test R²: {results['metrics']['test_r2']:.4f}")
    logger.info(f"✓ Selected {len(results['selected_features'])} features")

    return model_path, metadata_path


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train ML models for price prediction"
    )
    parser.add_argument(
        "--mode",
        choices=["final", "walk_forward"],
        default="final",
        help="Training mode: 'final' for train/validation/test split, "
             "'walk_forward' for walk-forward validation.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "MSFT", "GOOG"],
        help="Stock symbols to train on (default: AAPL MSFT GOOG)",
    )
    parser.add_argument(
        "--start",
        default="2015-01-01",
        help="Start date for historical data (default: 2015-01-01)",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=500,
        help="Training window size for walk-forward (default: 500)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=100,
        help="Test window size for walk-forward (default: 100)",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=50,
        help="Step size for walk-forward (default: 50)",
    )
    parser.add_argument(
        "--model-path",
        default="ai/models/pipeline_model.pkl",
        help="Path to save the trained pipeline model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose pipeline logging.",
    )

    args = parser.parse_args()

    if args.mode == "final":
        train_with_final_split(
            args.symbols,
            args.start,
            model_path=args.model_path,
            verbose=args.verbose,
        )
    elif args.mode == "walk_forward":
        train_with_walk_forward(
            args.symbols,
            args.start,
            args.train_size,
            args.test_size,
            args.step_size,
            model_path=args.model_path,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()

