"""Deterministic walk-forward backtest engine for Mini-GRP.

The demo data path is synthetic and explicitly labelled as such. Returns are
linked to the generated factor signal so the demo verifies the pipeline rather
than presenting independent random performance. Research callers must provide
point-in-time stock and price data and should never fall back to demo data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)


class BacktestError(RuntimeError):
    """Raised when a backtest cannot produce a valid investable result."""


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    rebalance_freq: str = "monthly"
    top_n: int = 20
    benchmark: str = "equal_weight_universe"
    initial_capital: float = 1_000_000.0
    transaction_cost: float = 0.001

    def validate(self) -> None:
        start = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date)
        if start >= end:
            raise ValueError("start_date must be earlier than end_date")
        if self.rebalance_freq not in {"monthly", "quarterly"}:
            raise ValueError("rebalance_freq must be 'monthly' or 'quarterly'")
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if not 0 <= self.transaction_cost < 0.1:
            raise ValueError("transaction_cost must be a decimal between 0 and 0.1")

    @property
    def periods_per_year(self) -> int:
        return 12 if self.rebalance_freq == "monthly" else 4


def generate_rebalance_dates(config: BacktestConfig) -> List[str]:
    """Return month-end or quarter-end rebalance dates inside the interval."""
    config.validate()
    # Offset objects work across pandas 1.5, 2.x, and 3.x. The string aliases
    # changed between releases and previously broke the deployed application.
    freq = (
        pd.offsets.MonthEnd()
        if config.rebalance_freq == "monthly"
        else pd.offsets.QuarterEnd(startingMonth=12)
    )
    dates = pd.date_range(config.start_date, config.end_date, freq=freq)
    return dates.strftime("%Y-%m-%d").tolist()


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std == 0 or not np.isfinite(std):
        return np.zeros_like(values, dtype=float)
    return (values - float(np.mean(values))) / std


def _period_months(rebalance_freq: str) -> int:
    return 1 if rebalance_freq == "monthly" else 3


def generate_mock_backtest_data(
    n_stocks: int = 100,
    n_periods: int = 60,
    start_date: str = "2020-01-01",
    rebalance_freq: str = "monthly",
    seed: int = 42,
    rebalance_dates: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Generate deterministic synthetic factor snapshots and holding windows.

    The generated return has a modest relationship with the latent five-factor
    signal. This is useful for regression tests and UI demonstrations only; it
    is not evidence of an investable strategy.
    """
    if n_stocks < 10:
        raise ValueError("n_stocks must be at least 10")
    if n_periods < 2 and rebalance_dates is None:
        raise ValueError("n_periods must be at least 2")

    if rebalance_dates is None:
        months = _period_months(rebalance_freq)
        start = pd.Timestamp(start_date) + pd.offsets.MonthEnd(0)
        dates = [
            (start + relativedelta(months=months * index)).strftime("%Y-%m-%d")
            for index in range(n_periods)
        ]
    else:
        dates = list(rebalance_dates)

    rng = np.random.default_rng(seed)
    codes = [f"{600000 + index * 7:06d}" for index in range(n_stocks)]
    names = [f"DemoStock_{index:03d}" for index in range(n_stocks)]
    industries = np.array(
        ["Bank", "Financials", "Healthcare", "Electronics", "Consumer", "Energy", "Technology", "Industrials"]
    )
    stock_industries = rng.choice(industries, size=n_stocks)

    persistent_value = rng.normal(0, 1, n_stocks)
    persistent_quality = rng.normal(0, 1, n_stocks)
    persistent_growth = rng.normal(0, 1, n_stocks)
    persistent_momentum = rng.normal(0, 1, n_stocks)
    persistent_expectation = rng.normal(0, 1, n_stocks)

    snapshots: List[pd.DataFrame] = []
    price_data: Dict[str, pd.DataFrame] = {}
    holding_months = _period_months(rebalance_freq)
    trading_days = 22 * holding_months

    for period_index, trade_date in enumerate(dates):
        drift = period_index / max(len(dates) - 1, 1)
        value = persistent_value + rng.normal(0, 0.35, n_stocks)
        quality = persistent_quality + rng.normal(0, 0.30, n_stocks)
        growth = persistent_growth + rng.normal(0, 0.45, n_stocks) + 0.10 * drift
        momentum = 0.65 * persistent_momentum + rng.normal(0, 0.65, n_stocks)
        expectation = persistent_expectation + rng.normal(0, 0.50, n_stocks)

        composite_signal = (
            0.25 * _zscore(value)
            + 0.25 * _zscore(quality)
            + 0.15 * _zscore(growth)
            + 0.15 * _zscore(momentum)
            + 0.20 * _zscore(expectation)
        )

        pe = np.exp(3.1 - 0.35 * value)
        pb = np.exp(0.8 - 0.25 * value + 0.10 * quality)
        market_log_return = rng.normal(0.006 * holding_months, 0.025 * np.sqrt(holding_months))
        total_log_returns = (
            market_log_return
            + 0.012 * holding_months * composite_signal
            + rng.normal(0, 0.035 * np.sqrt(holding_months), n_stocks)
        )

        snapshot = pd.DataFrame(
            {
                "period": trade_date,
                "code": codes,
                "name": names,
                "sw_industry_name": stock_industries,
                "market": "DEMO",
                "currency": "N/A",
                "data_source": "synthetic_demo",
                "is_mock": True,
                "pe_ttm": pe,
                "pb_lf": pb,
                "ps_ttm": np.exp(1.2 - 0.25 * value + rng.normal(0, 0.20, n_stocks)),
                "ev_ebitda": np.exp(2.4 - 0.25 * value + rng.normal(0, 0.20, n_stocks)),
                "dividend_yield": np.clip(2.0 + 0.8 * value + rng.normal(0, 0.8, n_stocks), 0, None),
                "roe_deducted": 10 + 6 * quality + rng.normal(0, 2, n_stocks),
                "roa": 5 + 3 * quality + rng.normal(0, 1.5, n_stocks),
                "gross_margin": 30 + 8 * quality + rng.normal(0, 4, n_stocks),
                "net_margin": 12 + 5 * quality + rng.normal(0, 3, n_stocks),
                "debt_to_equity": np.clip(70 - 15 * quality + rng.normal(0, 12, n_stocks), 1, None),
                "revenue_yoy": 12 + 12 * growth + rng.normal(0, 6, n_stocks),
                "profit_yoy": 10 + 16 * growth + rng.normal(0, 8, n_stocks),
                "fcf_yield": 3 + 1.5 * growth + rng.normal(0, 1, n_stocks),
                "return_1m": 4 * momentum + rng.normal(0, 5, n_stocks),
                "return_3m": 8 * momentum + rng.normal(0, 8, n_stocks),
                "return_12m": 12 + 15 * momentum + rng.normal(0, 12, n_stocks),
                "sue": expectation + rng.normal(0, 0.35, n_stocks),
                "eps_revision": 0.15 * expectation + rng.normal(0, 0.08, n_stocks),
                "rating_revision": 0.25 * expectation + rng.normal(0, 0.12, n_stocks),
            }
        )
        snapshots.append(snapshot)

        window_start = pd.Timestamp(trade_date) + pd.offsets.BDay(1)
        index = pd.bdate_range(window_start, periods=trading_days + 1)
        prices: Dict[str, np.ndarray] = {}
        for code, total_log_return in zip(codes, total_log_returns):
            daily = rng.normal(0, 0.012, trading_days)
            daily += (total_log_return - float(daily.sum())) / trading_days
            prices[code] = 100.0 * np.exp(np.r_[0.0, np.cumsum(daily)])
        price_data[trade_date] = pd.DataFrame(prices, index=index)

    logger.info("Generated synthetic demo data: %d periods x %d stocks", len(dates), n_stocks)
    return pd.concat(snapshots, ignore_index=True), price_data


def _spearman_rank_correlation(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat([left, right], axis=1).dropna()
    if len(aligned) < 10:
        return float("nan")
    return float(aligned.iloc[:, 0].rank().corr(aligned.iloc[:, 1].rank()))


def _equal_weights(codes: Sequence[str]) -> Dict[str, float]:
    unique = list(dict.fromkeys(codes))
    if not unique:
        return {}
    weight = 1.0 / len(unique)
    return {code: weight for code in unique}


def _turnover_and_cost(
    previous: Sequence[str], selected: Sequence[str], transaction_cost: float
) -> Tuple[float, float, float]:
    previous_weights = _equal_weights(previous)
    target_weights = _equal_weights(selected)
    universe = set(previous_weights) | set(target_weights)
    traded_notional = sum(
        abs(target_weights.get(code, 0.0) - previous_weights.get(code, 0.0))
        for code in universe
    )
    turnover = traded_notional if not previous_weights else traded_notional / 2.0
    cost = traded_notional * transaction_cost
    return turnover, traded_notional, cost


def run_single_period(
    trade_date: str,
    next_trade_date: str,
    stock_data: pd.DataFrame,
    price_data: Dict[str, pd.DataFrame],
    prev_holdings: Optional[List[str]] = None,
    config: BacktestConfig = BacktestConfig(),
) -> Dict[str, Any]:
    """Score one point-in-time snapshot and hold through its price window."""
    from core.factor_engine import calculate_factors
    from core.scoring_engine import composite_score, get_top_picks, rank_within_industry, score_by_dimension

    period_df = stock_data.loc[stock_data["period"] == trade_date].copy()
    if period_df.empty:
        raise BacktestError(f"No point-in-time snapshot for {trade_date}")
    if trade_date not in price_data:
        raise BacktestError(f"No holding-period price window for {trade_date}")

    factor_df = calculate_factors(period_df)
    scored = score_by_dimension(factor_df)
    scored = composite_score(scored)
    scored = rank_within_industry(scored)
    top_picks = get_top_picks(scored, n=min(config.top_n, len(scored)))
    selected = top_picks["code"].astype(str).tolist()
    if not selected:
        raise BacktestError(f"No holdings selected for {trade_date}")

    price_window = price_data[trade_date].sort_index()
    if len(price_window) < 2:
        raise BacktestError(f"Price window for {trade_date} has fewer than two observations")
    period_returns = price_window.iloc[-1] / price_window.iloc[0] - 1.0
    period_returns = period_returns.replace([np.inf, -np.inf], np.nan).dropna()
    selected_returns = period_returns.reindex(selected).dropna()
    if selected_returns.empty:
        raise BacktestError(f"Selected holdings have no returns for {trade_date}")

    gross_return = float(selected_returns.mean())
    benchmark_return = float(period_returns.mean())
    turnover, traded_notional, cost = _turnover_and_cost(
        prev_holdings or [], selected, config.transaction_cost
    )
    net_return = gross_return - cost

    forward_returns = scored["code"].astype(str).map(period_returns)
    ic = _spearman_rank_correlation(scored["composite_score"], forward_returns)

    return {
        "trade_date": trade_date,
        "next_trade_date": next_trade_date,
        "selected_stocks": selected,
        "scores": top_picks,
        "period_return": net_return * 100.0,
        "gross_return": gross_return * 100.0,
        "benchmark_return": benchmark_return * 100.0,
        "excess_return": (net_return - benchmark_return) * 100.0,
        "turnover": turnover,
        "traded_notional": traded_notional,
        "transaction_cost_paid": cost * 100.0,
        "ic": ic,
        "num_stocks": len(selected_returns),
    }


def run_backtest(
    config: BacktestConfig = BacktestConfig(),
    stock_data: Optional[pd.DataFrame] = None,
    price_data: Optional[Dict[str, pd.DataFrame]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    demo_seed: int = 42,
    demo_n_stocks: int = 100,
) -> pd.DataFrame:
    """Run a deterministic demo or caller-supplied research backtest."""
    config.validate()
    rebalance_dates = generate_rebalance_dates(config)
    if len(rebalance_dates) < 2:
        raise BacktestError("At least two rebalance dates are required")

    if (stock_data is None) != (price_data is None):
        raise ValueError("stock_data and price_data must be provided together")
    if stock_data is None and price_data is None:
        stock_data, price_data = generate_mock_backtest_data(
            n_stocks=demo_n_stocks,
            rebalance_freq=config.rebalance_freq,
            seed=demo_seed,
            rebalance_dates=rebalance_dates,
        )
        data_mode = "synthetic_demo"
    else:
        data_mode = "research"

    assert stock_data is not None
    assert price_data is not None
    required = {"period", "code", "name", "sw_industry_name"}
    missing = required - set(stock_data.columns)
    if missing:
        raise ValueError(f"stock_data is missing required columns: {sorted(missing)}")

    records: List[Dict[str, Any]] = []
    previous_holdings: List[str] = []
    portfolio_nav = 1.0
    benchmark_nav = 1.0

    for index, trade_date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[index + 1]
        if progress_callback:
            progress_callback(index + 1, len(rebalance_dates) - 1, f"Backtest {trade_date}")
        result = run_single_period(
            trade_date,
            next_date,
            stock_data,
            price_data,
            previous_holdings,
            config,
        )
        portfolio_nav *= 1.0 + result["period_return"] / 100.0
        benchmark_nav *= 1.0 + result["benchmark_return"] / 100.0
        if not np.isfinite(portfolio_nav) or not np.isfinite(benchmark_nav):
            raise BacktestError(f"Non-finite NAV at {trade_date}")

        records.append(
            {
                **{key: value for key, value in result.items() if key != "scores"},
                "portfolio_nav": portfolio_nav,
                "benchmark_nav": benchmark_nav,
                "excess_nav": portfolio_nav / benchmark_nav,
                "cumulative_return": (portfolio_nav - 1.0) * 100.0,
                "benchmark_cumulative": (benchmark_nav - 1.0) * 100.0,
                "excess_cumulative": (portfolio_nav / benchmark_nav - 1.0) * 100.0,
                "rebalance_freq": config.rebalance_freq,
                "data_mode": data_mode,
            }
        )
        previous_holdings = result["selected_stocks"]

    results = pd.DataFrame(records)
    if results.empty or int(results["num_stocks"].sum()) == 0:
        raise BacktestError("Backtest produced zero holdings")
    if results[["portfolio_nav", "benchmark_nav"]].isna().any().any():
        raise BacktestError("Backtest NAV contains NaN")
    return results


def _infer_periods_per_year(results: pd.DataFrame) -> int:
    if "rebalance_freq" in results and not results["rebalance_freq"].empty:
        return 4 if results["rebalance_freq"].iloc[0] == "quarterly" else 12
    dates = pd.to_datetime(results["trade_date"])
    if len(dates) > 1 and dates.diff().dropna().dt.days.median() > 60:
        return 4
    return 12


def analyze_backtest_results(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate frequency-aware performance and hard-gate diagnostics."""
    if results_df.empty:
        raise BacktestError("Cannot analyze an empty backtest")
    periods_per_year = _infer_periods_per_year(results_df)
    periods = len(results_df)
    years = periods / periods_per_year

    portfolio_nav = float(results_df["portfolio_nav"].iloc[-1])
    benchmark_nav = float(results_df["benchmark_nav"].iloc[-1])
    total_return = portfolio_nav - 1.0
    benchmark_return = benchmark_nav - 1.0
    excess_return = portfolio_nav / benchmark_nav - 1.0
    annualized_return = portfolio_nav ** (1.0 / years) - 1.0
    annualized_benchmark = benchmark_nav ** (1.0 / years) - 1.0
    annualized_excess = (portfolio_nav / benchmark_nav) ** (1.0 / years) - 1.0

    excess_period_returns = results_df["excess_return"] / 100.0
    excess_std = float(excess_period_returns.std(ddof=1))
    sharpe = (
        float(excess_period_returns.mean()) / excess_std * np.sqrt(periods_per_year)
        if periods > 1 and excess_std > 0
        else float("nan")
    )

    nav = pd.Series([1.0, *results_df["portfolio_nav"].astype(float).tolist()])
    drawdown = nav / nav.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else float("nan")

    ics = results_df["ic"].dropna().astype(float)
    ic_std = float(ics.std(ddof=1)) if len(ics) > 1 else float("nan")
    avg_ic = float(ics.mean()) if len(ics) else float("nan")
    icir = avg_ic / ic_std * np.sqrt(periods_per_year) if np.isfinite(ic_std) and ic_std > 0 else float("nan")

    def optional_round(value: float, digits: int) -> Optional[float]:
        return round(value, digits) if np.isfinite(value) else None

    return {
        "total_return": round(total_return * 100.0, 2),
        "annualized_return": round(annualized_return * 100.0, 2),
        "benchmark_return": round(benchmark_return * 100.0, 2),
        "annualized_benchmark": round(annualized_benchmark * 100.0, 2),
        "excess_return": round(excess_return * 100.0, 2),
        "annualized_excess": round(annualized_excess * 100.0, 2),
        "sharpe_ratio": optional_round(sharpe, 3),
        "max_drawdown": round(max_drawdown * 100.0, 2),
        "calmar_ratio": optional_round(calmar, 3),
        "win_rate": round(float((results_df["excess_return"] > 0).mean()) * 100.0, 1),
        "avg_ic": optional_round(avg_ic, 4),
        "icir": optional_round(icir, 3),
        "avg_turnover": round(float(results_df["turnover"].mean()), 3),
        "total_transaction_cost": round(float(results_df["transaction_cost_paid"].sum()), 3),
        "num_periods": periods,
        "trade_count": int((results_df["traded_notional"] > 0).sum()),
        "periods_per_year": periods_per_year,
        "data_mode": str(results_df["data_mode"].iloc[0]) if "data_mode" in results_df else "unknown",
    }


def generate_backtest_report(
    results_df: pd.DataFrame,
    analysis: Dict[str, Any],
    output_dir: str = "./output/backtest",
) -> Dict[str, str]:
    """Write reproducible CSV/JSON artifacts and optional diagnostic plots."""
    if results_df.empty:
        raise BacktestError("Cannot report an empty backtest")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    periods = results_df.drop(columns=["selected_stocks"], errors="ignore").copy()
    periods_path = output / "periods.csv"
    periods.to_csv(periods_path, index=False, encoding="utf-8-sig")

    equity_path = output / "equity.csv"
    results_df[["trade_date", "portfolio_nav", "benchmark_nav", "excess_nav"]].to_csv(
        equity_path, index=False, encoding="utf-8-sig"
    )

    holdings_rows = [
        {"trade_date": row.trade_date, "code": code, "weight": 1.0 / len(row.selected_stocks)}
        for row in results_df.itertuples()
        for code in row.selected_stocks
    ]
    holdings_path = output / "holdings.csv"
    pd.DataFrame(holdings_rows).to_csv(holdings_path, index=False, encoding="utf-8-sig")

    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_mode": analysis.get("data_mode"),
        "num_periods": analysis.get("num_periods"),
        "trade_count": analysis.get("trade_count"),
        "artifacts": [periods_path.name, equity_path.name, holdings_path.name, metrics_path.name],
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    paths = {
        "periods": str(periods_path),
        "equity": str(equity_path),
        "holdings": str(holdings_path),
        "metrics": str(metrics_path),
        "manifest": str(manifest_path),
    }

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(pd.to_datetime(results_df["trade_date"]), results_df["portfolio_nav"], label="Portfolio")
        ax.plot(pd.to_datetime(results_df["trade_date"]), results_df["benchmark_nav"], label="Benchmark")
        ax.set_title("Synthetic Demo Equity Curve" if analysis.get("data_mode") == "synthetic_demo" else "Equity Curve")
        ax.set_ylabel("NAV")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        chart_path = output / "equity_curve.png"
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        paths["equity_curve"] = str(chart_path)
    except ImportError:
        logger.info("matplotlib not installed; skipping equity chart")

    return paths


if __name__ == "__main__":
    demo_config = BacktestConfig(start_date="2022-01-01", end_date="2024-12-31")
    demo_results = run_backtest(demo_config)
    demo_analysis = analyze_backtest_results(demo_results)
    print(json.dumps(demo_analysis, ensure_ascii=False, indent=2))
