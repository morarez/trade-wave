import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional

from ai.train_model import train_with_final_split, train_with_walk_forward
from backtest import run_all_backtests
import api


def parse_symbol_list(symbol_string: str) -> Optional[List[str]]:
    if not symbol_string:
        return None
    return [part.strip().upper() for part in symbol_string.replace(",", " ").split() if part.strip()]


def parse_model_specs(model_strings: List[str]) -> List[Dict[str, object]]:
    models = []
    if not model_strings:
        return models
    for model_spec in model_strings:
        if "=" in model_spec:
            name, path = [part.strip() for part in model_spec.split("=", 1)]
        else:
            path = model_spec.strip()
            name = Path(path).stem
        if path:
            models.append({"name": name, "path": path, "threshold": 0.001})
    return models


def print_dataframe_summary(summary):
    print("\nBacktest summary:")
    for strategy, row in summary.iterrows():
        print(f"\n- {strategy}")
        for label, value in row.items():
            print(f"    {label}: {value:.4f}" if isinstance(value, float) else f"    {label}: {value}")


def train_command(args):
    logging.basicConfig(level=logging.INFO)
    if args.mode == "final":
        train_with_final_split(
            symbols=args.symbols,
            start=args.start,
            model_path=args.model_path,
            verbose=args.verbose,
        )
    else:
        train_with_walk_forward(
            symbols=args.symbols,
            start=args.start,
            train_size=args.train_size,
            test_size=args.test_size,
            step_size=args.step_size,
            model_path=args.model_path,
            verbose=args.verbose,
        )


def backtest_command(args):
    symbols = parse_symbol_list(args.symbols) if args.symbols else None
    strategy_names = [s.strip() for s in args.strategies.split(",") if s.strip()] if args.strategies else None
    ai_models = parse_model_specs(args.model or [])

    portfolios, summary, per_symbol, _, _, _ = run_all_backtests(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        cash=args.cash,
        interval=args.interval,
        strategy_names=strategy_names,
        ai_models=ai_models,
    )

    print_dataframe_summary(summary)
    return portfolios, summary, per_symbol


def serve_command(args):
    api.run_server(host=args.host, port=args.port, debug=args.debug)


def main():
    parser = argparse.ArgumentParser(
        description="Trade Wave AI training and backtest comparison CLI"
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train an AI model")
    train_parser.add_argument("--mode", choices=["final", "walk_forward"], default="final")
    train_parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "GOOG"])
    train_parser.add_argument("--start", default="2015-01-01")
    train_parser.add_argument("--model-path", default="ai/models/pipeline_model.pkl")
    train_parser.add_argument("--train-size", type=int, default=500)
    train_parser.add_argument("--test-size", type=int, default=100)
    train_parser.add_argument("--step-size", type=int, default=50)
    train_parser.add_argument("--verbose", action="store_true")
    train_parser.set_defaults(func=train_command)

    backtest_parser = subparsers.add_parser("backtest", help="Run backtests for strategies and AI models")
    backtest_parser.add_argument("--symbols", default="AAPL,MSFT,GOOG")
    backtest_parser.add_argument("--start", default="2024-01-01")
    backtest_parser.add_argument("--end", default=None)
    backtest_parser.add_argument("--cash", type=float, default=10000)
    backtest_parser.add_argument("--interval", default="1d")
    backtest_parser.add_argument("--strategies", default="")
    backtest_parser.add_argument("--model", action="append", help="AI model specification as name=path or path")
    backtest_parser.set_defaults(func=backtest_command)

    compare_parser = subparsers.add_parser("compare", help="Alias for backtest")
    compare_parser.add_argument("--symbols", default="AAPL,MSFT,GOOG")
    compare_parser.add_argument("--start", default="2024-01-01")
    compare_parser.add_argument("--end", default=None)
    compare_parser.add_argument("--cash", type=float, default=10000)
    compare_parser.add_argument("--interval", default="1d")
    compare_parser.add_argument("--strategies", default="")
    compare_parser.add_argument("--model", action="append", help="AI model specification as name=path or path")
    compare_parser.set_defaults(func=backtest_command)

    serve_parser = subparsers.add_parser("serve", help="Run the Flask API server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=5000)
    serve_parser.add_argument("--debug", action="store_true")
    serve_parser.set_defaults(func=serve_command)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
