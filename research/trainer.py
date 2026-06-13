"""Leakage-aware walk-forward training for the Mini-GRP nonlinear overlay."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analytics.factor_research import extract_model_importance, run_factor_research
from core.main import score_universe
from research.storage import FACTOR_COLUMNS, ResearchStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    min_train_dates: int = 8
    validation_dates: int = 6
    overlay_weight: float = 0.15
    approval_min_ic: float = 0.0
    approval_min_spread: float = 0.0


MODEL_FEATURES = [
    "value_score",
    "quality_score",
    "growth_score",
    "momentum_score",
    *[f"{column}_z" for column in FACTOR_COLUMNS],
]


def _rank_ic(prediction: pd.Series, actual: pd.Series) -> float:
    valid = prediction.notna() & actual.notna()
    if valid.sum() < 20:
        return float("nan")
    return float(prediction[valid].rank().corr(actual[valid].rank()))


def _top_bottom_spread(prediction: pd.Series, actual: pd.Series) -> float:
    valid = prediction.notna() & actual.notna()
    if valid.sum() < 20:
        return float("nan")
    ranked = prediction[valid].rank(pct=True)
    top = actual[valid][ranked >= 0.8].mean()
    bottom = actual[valid][ranked <= 0.2].mean()
    return float(top - bottom)


def prepare_training_panel(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Score each historical cross-section independently and attach ML labels."""
    labelled = snapshots.dropna(subset=["forward_return"]).copy()
    if labelled.empty:
        raise ValueError("No labelled snapshots are available for model training")
    panels = []
    for snapshot_date, frame in labelled.groupby("snapshot_date", sort=True):
        universe = frame.rename(columns={"industry": "sw_industry_name"}).copy()
        universe["currency"] = "CNY"
        universe["data_source"] = "tushare_history"
        universe["is_mock"] = False
        universe["as_of_date"] = str(snapshot_date)
        universe["expectation_source"] = "unavailable"
        universe["factor_coverage"] = universe[FACTOR_COLUMNS].notna().mean(axis=1)
        scored, _ = score_universe(universe, top_n=min(20, len(universe)))
        scored["snapshot_date"] = str(snapshot_date)
        scored["forward_return"] = frame.set_index("code").reindex(scored["code"])["forward_return"].values
        scored["target_rank"] = scored["forward_return"].rank(pct=True) * 100.0
        panels.append(scored)
    panel = pd.concat(panels, ignore_index=True)
    for feature in MODEL_FEATURES:
        if feature not in panel:
            panel[feature] = np.nan
    usable = [feature for feature in MODEL_FEATURES if panel[feature].notna().mean() >= 0.20]
    if len(usable) < 3:
        raise ValueError(f"Insufficient model features after coverage checks: {usable}")
    panel.attrs["model_features"] = usable
    return panel


def _candidate_models() -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=120,
                        learning_rate=0.03,
                        max_depth=2,
                        min_samples_leaf=30,
                        subsample=0.8,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def walk_forward_validate(panel: pd.DataFrame, features: list[str], config: TrainingConfig) -> dict[str, Any]:
    dates = sorted(panel["snapshot_date"].astype(str).unique().tolist())
    if len(dates) < config.min_train_dates + 3:
        raise ValueError(
            f"At least {config.min_train_dates + 3} labelled snapshot dates are required; found {len(dates)}"
        )
    validation_dates = dates[-min(config.validation_dates, len(dates) - config.min_train_dates) :]
    summaries: dict[str, Any] = {}
    for model_name, model in _candidate_models().items():
        folds = []
        for validation_date in validation_dates:
            train_dates = [value for value in dates if value < validation_date]
            if len(train_dates) < config.min_train_dates:
                continue
            train = panel[panel["snapshot_date"].isin(train_dates)].dropna(subset=["target_rank"])
            test = panel[panel["snapshot_date"] == validation_date].dropna(subset=["target_rank"])
            model.fit(train[features], train["target_rank"])
            prediction = pd.Series(model.predict(test[features]), index=test.index)
            folds.append(
                {
                    "date": validation_date,
                    "rank_ic": _rank_ic(prediction, test["forward_return"]),
                    "top_bottom_spread": _top_bottom_spread(prediction, test["forward_return"]),
                    "samples": int(len(test)),
                }
            )
        fold_frame = pd.DataFrame(folds)
        summaries[model_name] = {
            "folds": folds,
            "mean_rank_ic": float(fold_frame["rank_ic"].mean()),
            "icir": float(fold_frame["rank_ic"].mean() / fold_frame["rank_ic"].std(ddof=1))
            if len(fold_frame) > 1 and fold_frame["rank_ic"].std(ddof=1) > 0
            else None,
            "mean_top_bottom_spread": float(fold_frame["top_bottom_spread"].mean()),
        }
    return summaries


def train_and_register(store: ResearchStore, config: TrainingConfig = TrainingConfig()) -> dict[str, Any]:
    snapshots = store.load_snapshots(labelled_only=True)
    panel = prepare_training_panel(snapshots)
    features = list(panel.attrs["model_features"])
    validation = walk_forward_validate(panel, features, config)
    selected_name = max(
        validation,
        key=lambda name: (
            np.nan_to_num(validation[name]["mean_rank_ic"], nan=-999.0),
            np.nan_to_num(validation[name]["mean_top_bottom_spread"], nan=-999.0),
        ),
    )
    final_model = _candidate_models()[selected_name]
    final_model.fit(panel[features], panel["target_rank"])

    factor_research = run_factor_research(panel, features)

    selected_metrics = validation[selected_name]
    approved = (
        selected_metrics["mean_rank_ic"] > config.approval_min_ic
        and selected_metrics["mean_top_bottom_spread"] > config.approval_min_spread
        and len(selected_metrics["folds"]) >= 3
    )
    status = "approved" if approved else "candidate"
    dates = sorted(panel["snapshot_date"].unique().tolist())
    bundle = {
        "model": final_model,
        "model_type": selected_name,
        "features": features,
        "overlay_weight": config.overlay_weight,
        "trained_from": dates[0],
        "trained_through": dates[-1],
    }
    metrics = {
        "selected_model": selected_name,
        "status": status,
        "snapshot_dates": len(dates),
        "training_rows": int(len(panel)),
        "validation": validation,
        "factor_metrics": factor_research.summary.replace({np.nan: None}).to_dict(orient="records"),
        "model_importance": extract_model_importance(bundle).to_dict(orient="records"),
    }
    version = store.save_model(bundle, metrics, features, status=status)
    metrics["version"] = version
    logger.info("Registered %s model %s", status, version)
    return metrics
