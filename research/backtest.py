"""Point-in-time backtesting over stored monthly research snapshots."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.main import score_universe
from research.storage import FACTOR_COLUMNS, ResearchStore


def _max_drawdown(nav: pd.Series) -> float:
    running_max = nav.cummax()
    return float((nav / running_max - 1.0).min())


def run_snapshot_backtest(
    store: ResearchStore,
    top_n: int = 20,
    transaction_cost: float = 0.001,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run a linear baseline backtest using only factors known at each date."""
    snapshots = store.load_snapshots(labelled_only=True)
    if snapshots.empty:
        raise ValueError("No labelled research snapshots are available for backtesting")

    periods = []
    previous_codes: set[str] = set()
    for snapshot_date, frame in snapshots.groupby("snapshot_date", sort=True):
        usable = frame.dropna(subset=["forward_return"]).copy()
        if len(usable) < max(10, top_n):
            continue
        universe = usable.rename(columns={"industry": "sw_industry_name"})
        universe["currency"] = "CNY"
        universe["data_source"] = "tushare_history"
        universe["is_mock"] = False
        universe["as_of_date"] = str(snapshot_date)
        universe["expectation_source"] = "unavailable"
        universe["factor_coverage"] = universe[FACTOR_COLUMNS].notna().mean(axis=1)
        scored, _ = score_universe(universe, top_n=min(top_n, len(universe)))
        selected = scored.sort_values("composite_score", ascending=False).head(top_n)
        selected_codes = set(selected["code"].astype(str))
        turnover = 1.0 if not previous_codes else 1.0 - len(selected_codes & previous_codes) / max(top_n, 1)
        gross_return = float(selected["forward_return"].mean())
        benchmark_return = float(usable["forward_return"].mean())
        net_return = gross_return - turnover * transaction_cost
        periods.append(
            {
                "snapshot_date": str(snapshot_date),
                "label_date": selected["label_date"].dropna().astype(str).max(),
                "gross_return": gross_return,
                "net_return": net_return,
                "benchmark_return": benchmark_return,
                "excess_return": net_return - benchmark_return,
                "turnover": turnover,
                "selected_count": len(selected),
                "selected_codes": ",".join(sorted(selected_codes)),
            }
        )
        previous_codes = selected_codes

    results = pd.DataFrame(periods)
    if results.empty:
        raise ValueError("Stored snapshots did not contain enough labelled stocks for backtesting")
    results["portfolio_nav"] = (1.0 + results["net_return"]).cumprod()
    results["benchmark_nav"] = (1.0 + results["benchmark_return"]).cumprod()
    period_count = len(results)
    annualization = 12.0 / period_count
    annualized_return = float(results["portfolio_nav"].iloc[-1] ** annualization - 1.0)
    annualized_benchmark = float(results["benchmark_nav"].iloc[-1] ** annualization - 1.0)
    excess = results["excess_return"]
    metrics = {
        "periods": period_count,
        "annualized_return": annualized_return,
        "annualized_benchmark": annualized_benchmark,
        "annualized_excess": annualized_return - annualized_benchmark,
        "max_drawdown": _max_drawdown(results["portfolio_nav"]),
        "information_ratio": float(excess.mean() / excess.std(ddof=1) * np.sqrt(12))
        if len(excess) > 1 and excess.std(ddof=1) > 0
        else None,
        "average_turnover": float(results["turnover"].mean()),
    }
    return results, metrics
