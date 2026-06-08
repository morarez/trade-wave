"""Example script demonstrating the new ML pipeline functionality.

This script shows:
1. Training a model with the final split approach
2. Training with walk-forward validation
3. Loading and using the trained model
4. Inspecting model metadata and feature importance
"""

import sys
import os
import logging
from datetime import datetime

# Ensure project root is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import json
from data_handler import get_yfinance_data
from ai.data import build_dataset
from ai.pipeline import TimeSeriesPipeline
from ai.predict import (
    load_pipeline_model,
    get_selected_features,
    get_model_metrics,
)


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def example_train_final_split():
    """Example 1: Train using final split approach."""
    print_header("Example 1: Train Model with Final Split")
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Fetch data
    print("\n1. Fetching historical data...")
    symbols = ["AAPL", "MSFT", "GOOG"]
    price_df = get_yfinance_data(symbols=symbols, start="2020-01-01")
    print(f"   ✓ Fetched {len(price_df)} days of data for {len(price_df.columns)} symbols")
    
    # Build features
    print("\n2. Building features and targets...")
    X, y = build_dataset(price_df)
    print(f"   ✓ Generated {X.shape[1]} features for {X.shape[0]} samples")
    print(f"   ✓ Features: {list(X.columns[:5])}... (showing first 5)")
    
    # Initialize pipeline
    print("\n3. Initializing pipeline...")
    pipeline = TimeSeriesPipeline(
        models_dir="ai/models",
        n_features=20,
        variance_threshold=0.01,
        verbose=True,
    )
    print("   ✓ Pipeline initialized")
    
    # Train model
    print("\n4. Training final model...")
    results = pipeline.train_final_model(X, y, test_size=0.2)
    
    print("\n5. Saving model and metadata...")
    model_path, metadata_path = pipeline.save_model("ai/models/pipeline_model.pkl")
    print(f"   ✓ Model saved to {model_path}")
    print(f"   ✓ Metadata saved to {metadata_path}")
    
    # Display results
    print("\n6. Model Performance:")
    metrics = results["metrics"]
    print(f"   Train MSE: {metrics['train_mse']:.6f}")
    print(f"   Val MSE:   {metrics['val_mse']:.6f}")
    print(f"   Test MSE:  {metrics['test_mse']:.6f}")
    print(f"   Train R²:  {metrics['train_r2']:.4f}")
    print(f"   Val R²:    {metrics['val_r2']:.4f}")
    print(f"   Test R²:   {metrics['test_r2']:.4f}")
    
    print(f"\n7. Selected Features ({len(results['selected_features'])}):")
    for i, feat in enumerate(results['selected_features'][:10], 1):
        print(f"   {i}. {feat}")
    if len(results['selected_features']) > 10:
        print(f"   ... and {len(results['selected_features']) - 10} more")
    
    return model_path


def example_walk_forward():
    """Example 2: Train using walk-forward validation."""
    print_header("Example 2: Walk-Forward Validation")
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Fetch data
    print("\n1. Fetching historical data...")
    symbols = ["AAPL", "MSFT"]
    price_df = get_yfinance_data(symbols=symbols, start="2020-01-01")
    print(f"   ✓ Fetched {len(price_df)} days of data")
    
    # Build features
    print("\n2. Building features and targets...")
    X, y = build_dataset(price_df)
    print(f"   ✓ Generated {X.shape[0]} samples with {X.shape[1]} features")
    
    # Initialize pipeline
    print("\n3. Initializing pipeline...")
    pipeline = TimeSeriesPipeline(
        models_dir="ai/models",
        n_features=15,
        variance_threshold=0.01,
        verbose=True,
    )
    
    # Perform walk-forward validation
    print("\n4. Running walk-forward validation...")
    print("   (with train_size=400, test_size=100, step_size=50)")
    wf_results = pipeline.walk_forward_validation(
        X, y, train_size=400, test_size=100, step_size=50
    )
    
    print("\n5. Walk-Forward Results:")
    print(f"   Overall MSE: {wf_results['overall_mse']:.6f}")
    print(f"   Overall MAE: {wf_results['overall_mae']:.6f}")
    print(f"   Overall R²:  {wf_results['overall_r2']:.4f}")
    print(f"   Number of folds: {wf_results['n_folds']}")
    
    print("\n6. Per-Fold Metrics:")
    for fold in wf_results['folds'][:3]:
        print(f"   Fold {fold['fold']}: MSE={fold['mse']:.6f}, "
              f"MAE={fold['mae']:.6f}, R²={fold['r2']:.4f} "
              f"(n_train={fold['n_train']}, n_test={fold['n_test']})")
    if len(wf_results['folds']) > 3:
        print(f"   ... ({len(wf_results['folds']) - 3} more folds)")


def example_load_and_predict():
    """Example 3: Load trained model and make predictions."""
    print_header("Example 3: Load Model and Make Predictions")
    
    model_path = "ai/models/pipeline_model.pkl"
    
    if not os.path.exists(model_path):
        print(f"\n⚠ Model not found at {model_path}")
        print("   Run example_train_final_split() first")
        return
    
    # Load model
    print(f"\n1. Loading model from {model_path}...")
    pipeline, metadata = load_pipeline_model(model_path)
    print("   ✓ Model loaded successfully")
    
    # Get selected features
    print("\n2. Getting model information...")
    selected_features = get_selected_features(model_path)
    print(f"   ✓ Selected features: {len(selected_features)} total")
    print(f"   ✓ First 5 features: {selected_features[:5]}")
    
    # Get metrics
    print("\n3. Model Performance Metrics:")
    metrics = get_model_metrics(model_path)
    print(f"   Train MSE: {metrics.get('train_mse', 'N/A')}")
    print(f"   Val MSE:   {metrics.get('val_mse', 'N/A')}")
    print(f"   Test MSE:  {metrics.get('test_mse', 'N/A')}")
    
    # Generate sample predictions
    print("\n4. Making sample predictions...")
    from ai.data import features_for_series
    from data_handler import get_yfinance_data
    
    price_df = get_yfinance_data(symbols=["AAPL"], start="2024-01-01")
    price_series = price_df["AAPL"].tail(100)
    X = features_for_series(price_series)
    
    if not X.empty:
        X_selected = X[selected_features] if all(f in X.columns for f in selected_features) else X
        preds = pipeline.predict(X_selected)
        print(f"   ✓ Generated {len(preds)} predictions")
        print(f"   ✓ Prediction range: [{preds.min():.6f}, {preds.max():.6f}]")
        print(f"   ✓ Mean prediction: {preds.mean():.6f}")
        print(f"   ✓ Std prediction:  {preds.std():.6f}")
    else:
        print("   ⚠ Could not generate features from sample data")


def example_feature_importance():
    """Example 4: Inspect feature importance."""
    print_header("Example 4: Feature Importance Analysis")
    
    model_path = "ai/models/pipeline_model.pkl"
    metadata_path = model_path.replace(".pkl", "_metadata.json")
    
    if not os.path.exists(metadata_path):
        print(f"\n⚠ Metadata not found at {metadata_path}")
        print("   Run example_train_final_split() first")
        return
    
    # Load metadata
    print(f"\n1. Loading metadata from {metadata_path}...")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    print("   ✓ Metadata loaded")
    
    # Display timestamp
    print(f"\n2. Model Information:")
    print(f"   Timestamp: {metadata.get('timestamp', 'N/A')}")
    print(f"   Selected features: {metadata.get('n_selected_features', 0)}")
    
    # Show feature importance
    print(f"\n3. Top 10 Most Important Features:")
    feature_importance = metadata.get("feature_importance", {})
    if feature_importance:
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for i, (feat, importance) in enumerate(sorted_features[:10], 1):
            bar_length = int(importance * 50)
            bar = "█" * bar_length
            print(f"   {i:2d}. {feat:20s} {bar} {importance:.4f}")
    else:
        print("   ⚠ Feature importance data not available")
    
    # Show split sizes
    print(f"\n4. Data Split Sizes:")
    splits = metadata.get("split_sizes", {})
    total = sum(splits.values())
    for split_name, size in splits.items():
        pct = (size / total * 100) if total > 0 else 0
        print(f"   {split_name:12s}: {size:6d} ({pct:5.1f}%)")
    
    # Show metrics
    print(f"\n5. Performance Metrics:")
    metrics = metadata.get("metrics", {})
    for metric_name, value in metrics.items():
        print(f"   {metric_name:12s}: {value:.6f}")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Trade-Wave AI Pipeline - Examples & Demonstrations".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    import argparse
    parser = argparse.ArgumentParser(description="Run pipeline examples")
    parser.add_argument(
        "--example",
        choices=["all", "train", "walk", "load", "importance"],
        default="train",
        help="Which example to run",
    )
    
    args = parser.parse_args()
    
    try:
        if args.example in ["all", "train"]:
            example_train_final_split()
        
        if args.example in ["all", "walk"]:
            example_walk_forward()
        
        if args.example in ["all", "load"]:
            example_load_and_predict()
        
        if args.example in ["all", "importance"]:
            example_feature_importance()
        
        print("\n" + "=" * 70)
        print("  Examples completed successfully!")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
