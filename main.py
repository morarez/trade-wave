import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional

from ai.train_model import train_with_final_split, train_with_walk_forward
from backtest import run_all_backtests
import api
from config import load_config


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
    config = load_config()
    logging.basicConfig(level=logging.INFO)
    if args.mode == "final":
        train_with_final_split(
            symbols=args.symbols or config["train"]["symbols"],
            start=args.start or config["train"]["start"],
            model_path=args.model_path or config["train"]["model_path"],
            verbose=args.verbose or config["train"]["verbose"],
        )
    else:
        train_with_walk_forward(
            symbols=args.symbols or config["train"]["symbols"],
            start=args.start or config["train"]["start"],
            train_size=args.train_size if args.train_size is not None else config["train"]["train_size"],
            test_size=args.test_size if args.test_size is not None else config["train"]["test_size"],
            step_size=args.step_size if args.step_size is not None else config["train"]["step_size"],
            model_path=args.model_path or config["train"]["model_path"],
            verbose=args.verbose or config["train"]["verbose"],
        )


def backtest_command(args):
    config = load_config()
    symbols = parse_symbol_list(args.symbols) if args.symbols else None
    strategy_names = [s.strip() for s in args.strategies.split(",") if s.strip()] if args.strategies else None
    ai_models = parse_model_specs(args.model or [])

    portfolios, summary, per_symbol, _, _, _ = run_all_backtests(
        symbols=symbols or parse_symbol_list(config["backtest"]["symbols"]),
        start_date=args.start or config["backtest"]["start"],
        end_date=args.end if args.end is not None else config["backtest"]["end"],
        cash=args.cash if args.cash is not None else config["backtest"]["cash"],
        interval=args.interval or config["backtest"]["interval"],
        strategy_names=strategy_names or ([s.strip() for s in config["backtest"]["strategies"].split(",") if s.strip()] if config["backtest"]["strategies"] else None),
        ai_models=ai_models,
    )

    print_dataframe_summary(summary)
    return portfolios, summary, per_symbol


def serve_command(args):
    config = load_config()
    api.run_server(
        host=args.host or config["api"]["host"],
        port=args.port if args.port is not None else config["api"]["port"],
        debug=args.debug if args.debug is not None else config["api"]["debug"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Trade Wave AI training and backtest comparison CLI"
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train an AI model")
    train_parser.add_argument("--mode", choices=["final", "walk_forward"], default=None)
    train_parser.add_argument("--symbols", nargs="+", default=None)
    train_parser.add_argument("--start", default=None)
    train_parser.add_argument("--model-path", default=None)
    train_parser.add_argument("--train-size", type=int, default=None)
    train_parser.add_argument("--test-size", type=int, default=None)
    train_parser.add_argument("--step-size", type=int, default=None)
    train_parser.add_argument("--verbose", action="store_true")
    train_parser.set_defaults(func=train_command)

    backtest_parser = subparsers.add_parser("backtest", help="Run backtests for strategies and AI models")
    backtest_parser.add_argument("--symbols", default=None)
    backtest_parser.add_argument("--start", default=None)
    backtest_parser.add_argument("--end", default=None)
    backtest_parser.add_argument("--cash", type=float, default=None)
    backtest_parser.add_argument("--interval", default=None)
    backtest_parser.add_argument("--strategies", default=None)
    backtest_parser.add_argument("--model", action="append", help="AI model specification as name=path or path")
    backtest_parser.set_defaults(func=backtest_command)

    compare_parser = subparsers.add_parser("compare", help="Alias for backtest")
    compare_parser.add_argument("--symbols", default=None)
    compare_parser.add_argument("--start", default=None)
    compare_parser.add_argument("--end", default=None)
    compare_parser.add_argument("--cash", type=float, default=None)
    compare_parser.add_argument("--interval", default=None)
    compare_parser.add_argument("--strategies", default=None)
    compare_parser.add_argument("--model", action="append", help="AI model specification as name=path or path")
    compare_parser.set_defaults(func=backtest_command)

    serve_parser = subparsers.add_parser("serve", help="Run the Flask API server")
    serve_parser.add_argument("--host", default=None)
    serve_parser.add_argument("--port", type=int, default=None)
    serve_parser.add_argument("--debug", action="store_true")
    serve_parser.set_defaults(func=serve_command)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
