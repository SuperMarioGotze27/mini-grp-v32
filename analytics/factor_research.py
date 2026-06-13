"""Production single-factor diagnostics for point-in-time research panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorResearchConfig:
    """Configuration for cross-sectional factor diagnostics."""

    date_col: str = "snapshot_date"
    code_col: str = "code"
    return_col: str = "forward_return"
    n_quantiles: int = 5
    min_cross_section: int = 20
    periods_per_year: int = 12
    max_decay_lag: int = 6


@dataclass
class FactorResearchResult:
    """Tables produced by a complete factor research run."""

    summary: pd.DataFrame
    ic_series: pd.DataFrame
    quantile_returns: pd.DataFrame
    factor_correlation: pd.DataFrame
    decay: pd.DataFrame


def _rank_ic(factor: pd.Series, forward_return: pd.Series, min_samples: int) -> float:
    valid = pd.DataFrame({"factor": factor, "return": forward_return}).dropna()
    if len(valid) < min_samples or valid["factor"].nunique() < 3:
        return float("nan")
    return float(valid["factor"].rank(method="average").corr(valid["return"].rank(method="average")))


def _quantile_means(
    factor: pd.Series,
    forward_return: pd.Series,
    n_quantiles: int,
    min_samples: int,
) -> pd.Series:
    valid = pd.DataFrame({"factor": factor, "return": forward_return}).dropna()
    if len(valid) < max(min_samples, n_quantiles * 2) or valid["factor"].nunique() < n_quantiles:
        return pd.Series(dtype=float)
    ranks = valid["factor"].rank(method="first")
    valid["quantile"] = pd.qcut(ranks, q=n_quantiles, labels=False) + 1
    return valid.groupby("quantile")["return"].mean()


def _annualized_return(period_returns: pd.Series, periods_per_year: int) -> float:
    clean = period_returns.dropna()
    if clean.empty:
        return float("nan")
    if (clean <= -1.0).any():
        return float(clean.mean() * periods_per_year)
    return float((1.0 + clean).prod() ** (periods_per_year / len(clean)) - 1.0)


def _average_rank_autocorrelation(
    panel: pd.DataFrame,
    factor: str,
    config: FactorResearchConfig,
) -> float:
    ranks_by_date: list[pd.Series] = []
    for _, frame in panel.groupby(config.date_col, sort=True):
        clean = frame[[config.code_col, factor]].dropna().drop_duplicates(config.code_col)
        if len(clean) < config.min_cross_section:
            continue
        ranks_by_date.append(clean.set_index(config.code_col)[factor].rank(pct=True))

    correlations = []
    for previous, current in zip(ranks_by_date, ranks_by_date[1:]):
        joined = pd.concat([previous.rename("previous"), current.rename("current")], axis=1).dropna()
        if len(joined) >= config.min_cross_section:
            correlations.append(joined["previous"].corr(joined["current"]))
    return float(np.nanmean(correlations)) if correlations else float("nan")


def _factor_correlation(
    panel: pd.DataFrame,
    factors: list[str],
    config: FactorResearchConfig,
) -> pd.DataFrame:
    matrices = []
    for _, frame in panel.groupby(config.date_col, sort=True):
        usable = frame[factors].dropna(axis=1, how="all")
        if len(usable) >= config.min_cross_section and usable.shape[1] > 1:
            matrices.append(usable.corr(method="spearman"))
    if not matrices:
        return pd.DataFrame(index=factors, columns=factors, dtype=float)
    total = sum(matrix.reindex(index=factors, columns=factors).fillna(0.0) for matrix in matrices)
    counts = sum(
        matrix.reindex(index=factors, columns=factors).notna().astype(float)
        for matrix in matrices
    )
    result = total.divide(counts.replace(0.0, np.nan))
    np.fill_diagonal(result.values, 1.0)
    return result


def _decay_profile(
    panel: pd.DataFrame,
    factors: list[str],
    config: FactorResearchConfig,
) -> pd.DataFrame:
    dates = sorted(panel[config.date_col].astype(str).unique().tolist())
    indexed = {
        date: frame.drop_duplicates(config.code_col).set_index(config.code_col)
        for date, frame in panel.assign(**{config.date_col: panel[config.date_col].astype(str)}).groupby(
            config.date_col,
            sort=True,
        )
    }
    rows = []
    for factor in factors:
        for lag in range(config.max_decay_lag + 1):
            values = []
            for index, date in enumerate(dates):
                target_index = index + lag
                if target_index >= len(dates):
                    continue
                exposures = indexed[date][factor]
                future_returns = indexed[dates[target_index]][config.return_col]
                values.append(_rank_ic(exposures, future_returns, config.min_cross_section))
            clean = pd.Series(values, dtype=float).dropna()
            rows.append(
                {
                    "factor": factor,
                    "lag_months": lag,
                    "mean_rank_ic": float(clean.mean()) if not clean.empty else np.nan,
                    "periods": int(len(clean)),
                }
            )
    return pd.DataFrame(rows)


def run_factor_research(
    panel: pd.DataFrame,
    factor_cols: Iterable[str],
    config: FactorResearchConfig = FactorResearchConfig(),
) -> FactorResearchResult:
    """Evaluate factors across dated cross-sections without look-ahead leakage."""
    required = {config.date_col, config.code_col, config.return_col}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Factor panel is missing required columns: {sorted(missing)}")
    factors = [factor for factor in dict.fromkeys(factor_cols) if factor in panel.columns]
    if not factors:
        raise ValueError("No requested factor columns are present in the research panel")

    labelled = panel.dropna(subset=[config.return_col]).copy()
    if labelled.empty:
        raise ValueError("No labelled rows are available for factor research")
    labelled[config.date_col] = labelled[config.date_col].astype(str)

    summary_rows: list[dict[str, Any]] = []
    ic_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []

    for factor in factors:
        for date, frame in labelled.groupby(config.date_col, sort=True):
            ic = _rank_ic(frame[factor], frame[config.return_col], config.min_cross_section)
            if np.isfinite(ic):
                ic_rows.append({"snapshot_date": str(date), "factor": factor, "rank_ic": ic})
            means = _quantile_means(
                frame[factor],
                frame[config.return_col],
                config.n_quantiles,
                config.min_cross_section,
            )
            for quantile, value in means.items():
                quantile_rows.append(
                    {
                        "snapshot_date": str(date),
                        "factor": factor,
                        "quantile": int(quantile),
                        "forward_return": float(value),
                    }
                )

        factor_ic = pd.DataFrame(ic_rows)
        factor_ic = factor_ic[factor_ic["factor"] == factor]["rank_ic"] if not factor_ic.empty else pd.Series(dtype=float)
        factor_quantiles = pd.DataFrame(quantile_rows)
        if not factor_quantiles.empty:
            factor_quantiles = factor_quantiles[factor_quantiles["factor"] == factor]
        mean_ic = float(factor_ic.mean()) if not factor_ic.empty else np.nan
        ic_std = float(factor_ic.std(ddof=1)) if len(factor_ic) > 1 else np.nan
        icir = mean_ic / ic_std if np.isfinite(ic_std) and ic_std > 0 else np.nan

        mean_quantiles = (
            factor_quantiles.groupby("quantile")["forward_return"].mean()
            if not factor_quantiles.empty
            else pd.Series(dtype=float)
        )
        monotonicity = (
            float(pd.Series(mean_quantiles.index, dtype=float).corr(mean_quantiles.reset_index(drop=True), method="spearman"))
            if len(mean_quantiles) >= 3
            else np.nan
        )
        spread_by_date = pd.Series(dtype=float)
        if not factor_quantiles.empty:
            pivot = factor_quantiles.pivot(index="snapshot_date", columns="quantile", values="forward_return")
            if 1 in pivot and config.n_quantiles in pivot:
                spread_by_date = pivot[config.n_quantiles] - pivot[1]

        rank_autocorrelation = _average_rank_autocorrelation(labelled, factor, config)
        coverage = float(labelled[factor].notna().mean())
        positive_ratio = float((factor_ic > 0).mean()) if not factor_ic.empty else np.nan
        mean_spread = float(spread_by_date.mean()) if not spread_by_date.empty else np.nan
        spread_std = float(spread_by_date.std(ddof=1)) if len(spread_by_date) > 1 else np.nan
        spread_sharpe = (
            float(spread_by_date.mean() / spread_std * np.sqrt(config.periods_per_year))
            if np.isfinite(spread_std) and spread_std > 0
            else np.nan
        )
        passed = bool(
            np.isfinite(mean_ic)
            and mean_ic > 0.03
            and np.isfinite(icir)
            and icir > 0.5
            and positive_ratio > 0.55
            and np.isfinite(mean_spread)
            and mean_spread > 0
        )
        summary_rows.append(
            {
                "factor": factor,
                "periods": int(len(factor_ic)),
                "coverage": coverage,
                "mean_rank_ic": mean_ic,
                "ic_std": ic_std,
                "icir": icir,
                "ic_positive_ratio": positive_ratio,
                "mean_top_bottom_spread": mean_spread,
                "long_short_annualized": _annualized_return(spread_by_date, config.periods_per_year),
                "long_short_sharpe": spread_sharpe,
                "quantile_monotonicity": monotonicity,
                "rank_autocorrelation": rank_autocorrelation,
                "factor_turnover": 1.0 - rank_autocorrelation if np.isfinite(rank_autocorrelation) else np.nan,
                "passed": passed,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("mean_rank_ic", ascending=False).reset_index(drop=True)
    return FactorResearchResult(
        summary=summary,
        ic_series=pd.DataFrame(ic_rows),
        quantile_returns=pd.DataFrame(quantile_rows),
        factor_correlation=_factor_correlation(labelled, factors, config),
        decay=_decay_profile(labelled, factors, config),
    )


def extract_model_importance(bundle: dict[str, Any]) -> pd.DataFrame:
    """Extract normalized native importance from a fitted sklearn pipeline."""
    features = list(bundle.get("features", []))
    pipeline = bundle.get("model")
    if pipeline is None or not features:
        return pd.DataFrame(columns=["feature", "importance"])
    estimator = pipeline.named_steps.get("model", pipeline) if hasattr(pipeline, "named_steps") else pipeline
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        importance = np.abs(np.asarray(estimator.coef_, dtype=float).reshape(-1))
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    if len(importance) != len(features):
        return pd.DataFrame(columns=["feature", "importance"])
    total = float(np.nansum(importance))
    normalized = importance / total if total > 0 else importance
    return (
        pd.DataFrame({"feature": features, "importance": normalized})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
