"""Command line entry point for collection, training, and scheduled pipelines."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Iterable, Optional

from research.collector import CollectionConfig, collect_history
from research.backtest import run_snapshot_backtest
from research.storage import ResearchStore
from research.trainer import TrainingConfig, train_and_register


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mini-GRP v3.4 research pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "pipeline"):
        command = subparsers.add_parser(name)
        command.add_argument("--months", type=int, default=48)
        command.add_argument("--max-stocks", type=int, default=1500)
        command.add_argument("--forward-days", type=int, default=20)
        command.add_argument("--end-date")
        if name == "pipeline":
            command.add_argument("--overlay-weight", type=float, default=0.15)
    train = subparsers.add_parser("train")
    train.add_argument("--overlay-weight", type=float, default=0.15)
    backtest = subparsers.add_parser("backtest")
    backtest.add_argument("--top-n", type=int, default=20)
    backtest.add_argument("--transaction-cost", type=float, default=0.001)
    subparsers.add_parser("status")
    parser.add_argument("--database-url")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    store = ResearchStore(args.database_url)
    output = {}
    if args.command in {"collect", "pipeline"}:
        output["collection"] = collect_history(
            CollectionConfig(
                months=args.months,
                max_stocks=args.max_stocks,
                forward_days=args.forward_days,
                end_date=args.end_date,
            ),
            store,
        )
    if args.command in {"train", "pipeline"}:
        overlay_weight = getattr(args, "overlay_weight", 0.15)
        output["training"] = train_and_register(store, TrainingConfig(overlay_weight=overlay_weight))
    if args.command == "status":
        output = store.status()
        model = store.latest_model("approved")
        output["approved_model"] = None if model is None else {
            "version": model.version,
            "trained_through": model.trained_through,
            "metrics": model.metrics,
        }
    if args.command == "backtest":
        results, metrics = run_snapshot_backtest(
            store,
            top_n=args.top_n,
            transaction_cost=args.transaction_cost,
        )
        output = {"metrics": metrics, "periods": results.to_dict(orient="records")}
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
