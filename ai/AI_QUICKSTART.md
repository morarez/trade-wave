# Quick Start - AI Pipeline

Get up and running with the enhanced AI training pipeline in 5 minutes.

## Prerequisites

Ensure you have dependencies installed:
```bash
pip install -r requirements.txt
```

## Quick Training

### Option 1: Simple
```bash
python ai/train_model.py --mode final
```

**What it does:**
- Fetches AAPL, MSFT, GOOG data from 2015
- Selects top 20 features automatically
- Splits: 64% train, 16% validation, 20% test
- Trains LightGBM model
- Saves model + scaler + metadata

### Option 2: Advanced (Walk-Forward Validation)
```bash
python ai/train_model.py --mode walk_forward --train-size 500 --test-size 100
```

**What it does:**
- Runs time-series cross-validation
- Tests robustness across different periods
- Trains final model
- Reports per-fold metrics

## Check Results

### View Model Metrics
```bash
cat ai/models/pipeline_model_metadata.json
```

Output shows:
- Performance: MSE, MAE, R² for train/val/test
- Selected features list
- Feature importance scores
- Training timestamp

### Test Predictions
```python
from ai.predict import load_pipeline_model
from ai.data import features_for_series
from data_handler import get_yfinance_data

# Load model
pipeline, metadata = load_pipeline_model()

# Get recent data
df = get_yfinance_data(["AAPL"], start="2024-01-01")
X = features_for_series(df["AAPL"])

# Make predictions
predictions = pipeline.predict(X)
print(f"Predicted returns: {predictions[-5:]}")
```

## Use in Backtest

### Update Strategy
```python
from strategies.ai_strategy import strategy

# In your backtest
signals = strategy.generate_signals_pipeline(
    df,
    model_path="ai/models/pipeline_model.pkl",
    threshold=0.001
)
```

### Run Backtest
```bash
python backtest.py  # or your backtest script
```

## Customize Training

### Custom Symbols & Date Range
```bash
python ai/train_model.py \
  --mode final \
  --symbols AAPL MSFT GOOG TSLA \
  --start 2020-01-01
```

### More Features
```bash
# Edit ai/train_model.py, in train_with_final_split():
pipeline = TimeSeriesPipeline(
    n_features=30,  # Select 30 features instead of 20
    variance_threshold=0.001  # Lower threshold
)
```

### Walk-Forward Parameters
```bash
python ai/train_model.py \
  --mode walk_forward \
  --train-size 1000 \
  --test-size 200 \
  --step-size 100
```
## Examples

Run comprehensive examples:
```bash
# Train example
python ai/examples.py --example train

# Walk-forward example
python ai/examples.py --example walk

# Load and predict
python ai/examples.py --example load

# Feature importance analysis
python ai/examples.py --example importance

# All examples
python ai/examples.py --example all
```

## Troubleshooting

### "Model not found"
```bash
# Train first
python ai/train_model.py --mode final
```

### "Feature mismatch"
Ensure input data has same features as training data:
```python
from ai.predict import get_selected_features
features = get_selected_features()
X = X[features]  # Select same features
```

## Performance Tips

- **Speed:** Use `--mode final` for faster training
- **Robustness:** Use `--mode walk_forward` for production
- **Features:** Increase `n_features` for better fit (slower)
- **Data:** Use more symbols/longer history for stability
