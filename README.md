# trade-wave

Trade-Wave is a lightweight Python framework for backtesting systematic trading strategies using historical market data.

## Installation

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Run the backtest:

```bash
python main.py
```

## Usage

The default backtest runs for the symbol list configured in `backtest.py` and the strategies registered in `strategies/strategy_factory.py`.

To customize the run, update the call to `run_all_backtests()` in `main.py` or adjust the defaults in `backtest.py`:

* `symbols` - list of ticker symbols to backtest
* `start_date` / `end_date` - historical data range
* `cash` - starting portfolio cash amount
* `interval` - data interval such as `1d`, `5m`, or `1h`

## Output

`main.py` prints:

* overall strategy comparison statistics
* per-symbol summary statistics for each strategy

## AI Integration (LightGBM)

This project includes an optional AI-based strategy using LightGBM. The AI modules live in the `ai/` package and include utilities to build features, train a model, and produce model-based signals.

Quick start:

1. Train a model (this will save to `ai/models/lightgbm_model.pkl`):

```bash
python ai/train_model.py
```

2. Run the backtest including the AI strategy (registered in `strategies/strategy_factory.py`):

```bash
python main.py
```

Files added:

- `ai/data.py`: feature engineering helpers
- `ai/model.py`: training and model persistence
- `ai/predict.py`: load model and generate BUY/SELL/HOLD signals
- `ai/train_model.py`: example training script
- `strategies/ai_strategy.py`: wrapper so the AI model can be backtested alongside other strategies

Notes:

- The default AI model predicts next-period returns (regression) and converts predictions to signals via a threshold.
- Start with the default LightGBM parameters and iterate on features and training splits; use walk-forward validation for robust results.
