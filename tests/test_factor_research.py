from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from analytics.factor_research import extract_model_importance, run_factor_research


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(12)
    rows = []
    persistent = rng.normal(size=120)
    for month in range(14):
        predictive = 0.75 * persistent + rng.normal(scale=0.55, size=120)
        noise = rng.normal(size=120)
        forward_return = predictive * 0.025 + rng.normal(scale=0.012, size=120)
        for index in range(120):
            rows.append(
                {
                    "snapshot_date": f"2024{month + 1:02d}",
                    "code": f"{index:06d}",
                    "predictive": predictive[index],
                    "noise": noise[index],
                    "forward_return": forward_return[index],
                }
            )
    return pd.DataFrame(rows)


def test_factor_research_identifies_predictive_factor():
    result = run_factor_research(_panel(), ["predictive", "noise"])
    summary = result.summary.set_index("factor")

    assert summary.loc["predictive", "mean_rank_ic"] > 0.7
    assert summary.loc["predictive", "icir"] > 0.5
    assert summary.loc["predictive", "mean_top_bottom_spread"] > 0
    assert summary.loc["predictive", "quantile_monotonicity"] > 0.8
    assert bool(summary.loc["predictive", "passed"])
    assert summary.loc["predictive", "rank_autocorrelation"] > 0.4
    assert abs(summary.loc["noise", "mean_rank_ic"]) < 0.15
    assert result.factor_correlation.shape == (2, 2)
    assert set(result.decay["lag_months"]) == set(range(7))


def test_extract_model_importance_is_normalized():
    panel = _panel()
    model = Pipeline([("model", GradientBoostingRegressor(random_state=42))])
    model.fit(panel[["predictive", "noise"]], panel["forward_return"])

    importance = extract_model_importance(
        {"model": model, "features": ["predictive", "noise"]}
    )

    assert importance.iloc[0]["feature"] == "predictive"
    assert np.isclose(importance["importance"].sum(), 1.0)
