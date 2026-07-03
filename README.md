# Trade Wave

Trade Wave is an AI trading project that makes AI training the core workflow and uses backtesting as evaluation and comparison. It includes a CLI entrypoint, a Flask backtest API, and a Vite frontend for visualization.

## What it does

- Trains AI models for price prediction using the `ai/` pipeline.
- Runs historical backtests for benchmark strategies and AI models.
- Exposes a REST API for programmatic backtest comparison.

## Prerequisites

- Python 3.12+
- Node.js and npm

## Installation

1. Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Install frontend dependencies:

```bash
npm install
```

## Running locally

Use the CLI entrypoint in `main.py` for training, backtesting, comparing, and serving the API.

Start both the backend API server and the Vite frontend together with one command:

```bash
source venv/bin/activate
npm run dev
```

This will launch:

- the Python backend on http://127.0.0.1:5000
- the Vite frontend on the local URL shown in the terminal

If you want to run them separately, you can still use:

```bash
source venv/bin/activate
python main.py serve --host 127.0.0.1 --port 5000
```

and in another terminal:

```bash
npm run dev:frontend
```

Run a benchmark backtest:

```bash
python main.py backtest --symbols AAPL,MSFT --strategies sma_rsi,bollinger_rsi --start 2024-01-01 --end 2024-06-01
```

Compare benchmark strategies with one or more AI models:

```bash
python main.py compare --symbols AAPL,MSFT --strategies sma_rsi --model ai_model=ai/models/pipeline_model.pkl
```

In another terminal, start the frontend:

```bash
npm run dev
```

Open the Vite URL shown in the frontend terminal.

## Project structure

- `main.py` - CLI entrypoint for training, backtesting, comparison, and serving the API
- `api.py` - Flask API definition for `/api/backtest`
- `backtest.py` - backtest runner and strategy bridge
- `data_handler.py` - yfinance data fetcher
- `strategies/` - strategy implementations and registry
- `ai/` - AI model prediction and training helpers
- `frontend/` - React + TypeScript UI
- `tests/` - pytest tests for strategy and backtest behavior

## Backend API

The backend exposes a JSON POST API at:

- `/api/backtest`

Example request body:

```json
{
  "symbols": "AAPL, MSFT, GOOG",
  "strategies": "sma_rsi,bollinger_rsi",
  "models": ["ai_model=ai/models/pipeline_model.pkl"]
}
```

The response includes:

- `summary` — aggregated strategy metrics
- `per_symbol` — detailed per-symbol stats for each strategy

## AI

The AI model training pipeline lives in `ai/`.

Train a model with:

```bash
python main.py train --mode final --model-path ai/models/pipeline_model.pkl
```

Or use walk-forward validation:

```bash
python main.py train --mode walk_forward --train-size 500 --test-size 100 --step-size 50 --model-path ai/models/pipeline_model.pkl
```

Then include the trained AI model in backtest comparison:

```bash
python main.py backtest --symbols AAPL --strategies sma_rsi --model ai_model=ai/models/pipeline_model.pkl
```
