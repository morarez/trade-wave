# AI Pipeline Usage Guide

This guide explains the AI training pipeline with walk-forward validation, feature selection, and metadata management.

## Overview

The pipeline system (`ai/pipeline.py`) provides:

- **Walk-forward validation**: Proper time-series cross-validation using rolling windows
- **Feature selection**: Variance thresholding + mutual information scoring to select most predictive features
- **Feature scaling**: StandardScaler for normalized feature values
- **Metadata management**: Saves selected features, scaler parameters, feature importance, and metrics
- **Train/Validation/Test split**: Proper separation for model evaluation

## Training Modes

### 1. Final Split Mode

Trains on train set, optimizes on validation set, evaluates on test set.

```bash
python ai/train_model.py --mode final --symbols AAPL MSFT GOOG --start 2015-01-01
```

This will:
1. Fetch historical data
2. Build features using technical indicators
3. Select top 20 features by mutual information
4. Scale features using StandardScaler
5. Split data: 64% train, 16% validation, 20% test
6. Train LightGBM model
7. Save model, scaler, and metadata to `ai/models/pipeline_model.pkl`

### 2. Walk-Forward Validation Mode

Performs k-fold time-series cross-validation, then trains final model.

```bash
python ai/train_model.py \
  --mode walk_forward \
  --symbols AAPL MSFT GOOG \
  --start 2015-01-01 \
  --train-size 500 \
  --test-size 100 \
  --step-size 50
```

Parameters:
- `--train-size`: Number of samples per training window (default: 500)
- `--test-size`: Number of samples per test window (default: 100)
- `--step-size`: Step size for rolling window (default: 50)

This will:
1. Run walk-forward validation across multiple folds
2. Print metrics for each fold
3. Train final model on all data
4. Save all results

## Output Files

After training, the following files are created in `ai/models/`:

### `pipeline_model.pkl`
Contains:
- Trained LightGBM model
- StandardScaler (fitted on training data)
- Feature selector (variance threshold + SelectKBest)
- Metadata dictionary

### `pipeline_model_metadata.json`
Human-readable metadata including:
```json
{
  "timestamp": "2024-06-06T10:30:00",
  "selected_features": ["close", "ret_1", "sma_20", "rsi", ...],
  "n_selected_features": 20,
  "feature_importance": {
    "rsi": 0.25,
    "sma_20": 0.18,
    ...
  },
  "metrics": {
    "train_mse": 0.000123,
    "val_mse": 0.000156,
    "test_mse": 0.000178,
    "train_r2": 0.65,
    "val_r2": 0.62,
    "test_r2": 0.60
  },
  "split_sizes": {
    "train": 1500,
    "validation": 375,
    "test": 470
  }
}
```

## Using Trained Models

### Load Model and Metadata

```python
from ai.predict import load_pipeline_model, get_selected_features, get_model_metrics

# Load model
pipeline, metadata = load_pipeline_model("ai/models/pipeline_model.pkl")

# Get selected features
features = get_selected_features("ai/models/pipeline_model.pkl")
print(f"Model uses {len(features)} features: {features}")

# Get metrics
metrics = get_model_metrics("ai/models/pipeline_model.pkl")
print(f"Test MSE: {metrics['test_mse']:.6f}")
print(f"Test R²: {metrics['test_r2']:.4f}")
```

### Make Predictions

```python
from ai.data import features_for_series
from ai.predict import load_pipeline_model
import pandas as pd

# Prepare features for new data
X_new = features_for_series(price_series)

# Make predictions
pipeline, metadata = load_pipeline_model()
predictions = pipeline.predict(X_new)

# predictions is a numpy array of predicted returns
```

## Feature Engineering

The pipeline builds features from several indicators:

**Price-based features:**
- `ret_1`, `ret_5`, `ret_10`: Returns at different horizons
- `close`: Close price

**Technical indicators:**
- Simple Moving Averages (SMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Bollinger Bands
- Rolling Volatility
- ATR (Average True Range)
- ROC (Rate of Change)
- ADX (Average Directional Index)
- OBV (On-Balance Volume)
- Volume Features

The pipeline automatically:
1. Removes low-variance features (variance < 0.01)
2. Selects top 20 features by mutual information with target
3. Scales all features to mean=0, std=1

## Customizing the Pipeline

### Adjust Feature Selection

```python
from ai.pipeline import TimeSeriesPipeline

pipeline = TimeSeriesPipeline(
    n_features=30,              # Select 30 features instead of 20
    variance_threshold=0.001,   # Lower threshold for low-variance filter
    verbose=True
)
```

### Adjust Walk-Forward Parameters

```python
results = pipeline.walk_forward_validation(
    X, y,
    train_size=1000,  # Larger training window
    test_size=200,    # Larger test window
    step_size=100     # Larger steps
)
```

## Model Evaluation

### Walk-Forward Results

Each fold provides:
- MSE, MAE, R² on test window
- Train/test window sizes
- Index ranges for reproducibility

Overall summary includes mean metrics across all folds.

### Final Model Results

Evaluates on three separate sets:
- **Train Set**: Used to fit model
- **Validation Set**: Used for monitoring during development
- **Test Set**: Held-out set for final evaluation

Metrics provided:
- MSE: Mean squared error
- MAE: Mean absolute error
- R²: Coefficient of determination

## Best Practices

1. **Use walk-forward validation first** to understand model stability
2. **Check feature importance** to ensure non-trivial features are selected
3. **Monitor metrics across splits** - if test >> train, model may be overfitting
4. **Save metadata** for model lineage and reproducibility
5. **Version your models** with timestamps in metadata
6. **Re-train regularly** as market dynamics change

## Troubleshooting

### Model performs worse on test set

Check if:
- Walk-forward results are similar (indicates robustness)
- Selected features make sense
- Test period covers different market conditions

### Few features selected

Reduce `variance_threshold` or increase `n_features`:
```python
pipeline = TimeSeriesPipeline(
    n_features=30,
    variance_threshold=0.001
)
```

### Model predictions are NaN

Ensure:
- Input features don't have NaN values
- Selected features are present in prediction data
- Data is properly scaled

## Integration with Backtest

Update backtest to use the new pipeline model:

```python
from ai.data import features_for_series
from ai.predict import load_pipeline_model

# In backtest strategy
pipeline, metadata = load_pipeline_model()
features = features_for_series(price_series)
predictions = pipeline.predict(features)

# Use predictions for trading signals
```
