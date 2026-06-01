# Trade Wave

Trade Wave is a local backtesting project for systematic trading strategies. It combines a Python backend API with a TypeScript React frontend for an interactive local UI.

## What it does

- Runs historical backtests for registered trading strategies.
- Supports multiple symbols and strategy performance comparison.

## Prerequisites

- Python 3.12+ (a virtual environment is recommended)
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
cd frontend
npm install
```

## Running locally

The project is run with a Python backend API and a Vite frontend.

In one terminal, start the backend API:

```bash
source venv/bin/activate
python main.py
```

In another terminal, start the frontend:

```bash
cd frontend
npm run dev
```

Open the Vite URL shown in the frontend terminal.

## Project structure

- `main.py` - Python backend API server
- `backtest.py` - backtest runner and strategy bridge
- `data_handler.py` - yfinance data fetcher
- `strategies/` - strategy implementations and registry
- `ai/` - AI model prediction and training helpers
- `frontend/` - React + TypeScript UI
- `tests/` - pytest tests for strategy and backtest behavior

## Frontend details

The frontend is a Vite app in `frontend/`:

- `frontend/src/App.tsx` - main UI component
- `frontend/src/styles.css` - application styles
- `frontend/src/main.tsx` - React entrypoint


## Backend API

The backend exposes a JSON POST API at:

- `/api/backtest`

Payload example:

```json
{ "symbols": "AAPL, MSFT, GOOG" }
```

Response structure includes `summary` and `per_symbol` data tables.

## AI Integration

This project includes an AI strategy using LightGBM in the `ai/` package.

Train a model with:

```bash
python ai/train_model.py
```

The model file is expected at `ai/models/lightgbm_model.pkl`.
