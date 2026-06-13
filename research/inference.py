"""Governed model inference and bounded nonlinear score blending."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from research.storage import ModelRecord, ResearchStore


def apply_ml_overlay(
    scored: pd.DataFrame,
    model_record: Optional[ModelRecord] = None,
    store: Optional[ResearchStore] = None,
    allow_candidate: bool = False,
) -> pd.DataFrame:
    """Blend an approved, or explicitly allowed candidate, ML prediction."""
    record = model_record or (store.latest_model("approved") if store else None)
    if record is None:
        raise RuntimeError("No approved ML model is available in the model registry")
    if record.status != "approved" and not (allow_candidate and record.status == "candidate"):
        raise RuntimeError(
            f"Model {record.version} has status '{record.status}' and is not approved for inference"
        )
    result = scored.copy()
    features = list(record.bundle["features"])
    for feature in features:
        if feature not in result:
            result[feature] = np.nan
    raw_prediction = pd.Series(record.bundle["model"].predict(result[features]), index=result.index)
    result["ml_score"] = raw_prediction.rank(pct=True) * 100.0
    weight = float(record.bundle.get("overlay_weight", 0.15))
    weight = min(max(weight, 0.0), 0.30)
    result["linear_score"] = result["composite_score"]
    result["final_score"] = (1.0 - weight) * result["linear_score"] + weight * result["ml_score"]
    result["final_score"] = result["final_score"].round(2)
    result["final_rank"] = result["final_score"].rank(method="min", ascending=False).astype(int)
    if "sw_industry_name" in result:
        result["final_industry_rank"] = (
            result.groupby("sw_industry_name")["final_score"]
            .rank(method="min", ascending=False)
            .astype(int)
        )
    result["model_version"] = record.version
    result["model_status"] = record.status
    result["model_type"] = str(record.bundle.get("model_type", "unknown"))
    result["ml_overlay_weight"] = weight
    result["model_trained_through"] = record.trained_through
    return result.sort_values("final_score", ascending=False).reset_index(drop=True)
