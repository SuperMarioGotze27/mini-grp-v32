"""Command-line entry point for Mini-GRP v3.3."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from backtest.engine import (
    BacktestConfig,
    analyze_backtest_results,
    generate_backtest_report,
    run_backtest,
)
from core.factor_engine import calculate_factors
from core.scoring_engine import (
    ALL_DIMENSION_SCORE_COLS,
    composite_score,
    get_top_picks,
    rank_within_industry,
    score_by_dimension,
)
from data.unified_fetcher import DataSourceUnavailable, fetch_multi_market_data
from utils.mock import generate_mock_data

logger = logging.getLogger("mini_grp")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mini-GRP v3.3 quantitative screening and backtest system"
    )
    parser.add_argument("--mode", choices=["screen", "backtest"], default="screen")
    parser.add_argument(
        "--data-mode",
        choices=["demo", "research"],
        default="demo",
        help="demo uses deterministic synthetic data; research forbids synthetic fallback",
    )
    parser.add_argument("--market", nargs="+", choices=["cn", "us", "hk", "jp", "kr"], default=["cn"])
    parser.add_argument("--max-stocks", type=int, default=200)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--rebalance-freq", choices=["monthly", "quarterly"], default="monthly")
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--demo-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--no-cache", action="store_true")
    return parser


def fetch_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.data_mode == "demo":
        return generate_mock_data(args.max_stocks, seed=args.demo_seed)

    return fetch_multi_market_data(
        markets=[market.upper() for market in args.market],
        max_stocks_per_market=args.max_stocks,
        use_cache=not args.no_cache,
        allow_mock=False,
    )


def score_universe(data: pd.DataFrame, top_n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    if data is None or data.empty:
        raise ValueError("The stock universe is empty")
    factors = calculate_factors(data)
    scored = score_by_dimension(factors)
    scored = composite_score(scored)
    scored = rank_within_industry(scored)
    return scored, get_top_picks(scored, n=min(top_n, len(scored)))


def _write_screening_artifacts(
    scored: pd.DataFrame,
    top_picks: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "screening_universe.csv"
    top_path = output_dir / "top_picks.csv"
    manifest_path = output_dir / "screening_manifest.json"
    scored.to_csv(full_path, index=False, encoding="utf-8-sig")
    top_picks.to_csv(top_path, index=False, encoding="utf-8-sig")

    source = str(scored.get("data_source", pd.Series(["unknown"])).iloc[0])
    is_mock = bool(scored.get("is_mock", pd.Series([False])).all())
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_source": source,
        "is_mock": is_mock,
        "universe_size": len(scored),
        "top_n": len(top_picks),
        "dimension_columns": [c for c in ALL_DIMENSION_SCORE_COLS if c in scored],
        "artifacts": [full_path.name, top_path.name],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"universe": str(full_path), "top_picks": str(top_path), "manifest": str(manifest_path)}


def run_screening(args: argparse.Namespace) -> dict[str, str]:
    data = fetch_data(args)
    scored, top_picks = score_universe(data, args.top_n)
    paths = _write_screening_artifacts(scored, top_picks, Path(args.output_dir) / "screening")
    logger.info("Screening complete: %d stocks, top %d", len(scored), len(top_picks))
    print(top_picks.to_string(index=False))
    return paths


def run_backtest_mode(args: argparse.Namespace) -> dict[str, str]:
    if args.data_mode == "research":
        raise DataSourceUnavailable(
            "Research backtesting requires caller-supplied point-in-time snapshots and price windows. "
            "The CLI intentionally refuses to substitute synthetic history."
        )
    config = BacktestConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        rebalance_freq=args.rebalance_freq,
        top_n=args.top_n,
        transaction_cost=args.transaction_cost,
    )
    results = run_backtest(config=config, demo_seed=args.demo_seed, demo_n_stocks=args.max_stocks)
    analysis = analyze_backtest_results(results)
    paths = generate_backtest_report(results, analysis, str(Path(args.output_dir) / "backtest"))
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    return paths


def main(argv: Optional[Iterable[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.max_stocks < 10:
        raise ValueError("--max-stocks must be at least 10")
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")

    logger.info("Mini-GRP v3.3 | mode=%s | data_mode=%s", args.mode, args.data_mode)
    if args.mode == "backtest":
        run_backtest_mode(args)
    else:
        run_screening(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
