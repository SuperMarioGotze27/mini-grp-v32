from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import requests

from core.main import score_universe
from research.backtest import run_snapshot_backtest
from research.collector import CollectionConfig, TushareSnapshotCollector
from research.inference import apply_ml_overlay
from research.storage import FACTOR_COLUMNS, ResearchStore
from research.trainer import TrainingConfig, train_and_register


def _research_store(tmp_path) -> ResearchStore:
    return ResearchStore(f"sqlite:///{(tmp_path / 'research.db').as_posix()}")


def _snapshot(snapshot_date: str, seed: int, n_stocks: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "snapshot_date": snapshot_date,
            "label_date": snapshot_date,
            "code": [f"{index:06d}.SZ" for index in range(n_stocks)],
            "name": [f"Stock {index}" for index in range(n_stocks)],
            "industry": [f"Industry {index % 8}" for index in range(n_stocks)],
            "market": "CN",
        }
    )
    for column in FACTOR_COLUMNS:
        frame[column] = rng.normal(size=n_stocks)

    universe = frame.rename(columns={"industry": "sw_industry_name"}).copy()
    universe["currency"] = "CNY"
    universe["data_source"] = "test"
    universe["is_mock"] = False
    universe["as_of_date"] = snapshot_date
    universe["expectation_source"] = "unavailable"
    universe["factor_coverage"] = 1.0
    scored, _ = score_universe(universe, top_n=20)
    score_by_code = scored.set_index("code")["composite_score"]
    frame["forward_return"] = (
        frame["code"].map(score_by_code).astype(float) / 1000.0
        + rng.normal(scale=0.002, size=n_stocks)
    )
    return frame


def test_month_end_dates_respects_cutoff():
    open_dates = pd.bdate_range("2024-01-01", "2024-04-30").strftime("%Y%m%d").tolist()
    result = TushareSnapshotCollector.month_end_dates(open_dates, 3, "20240315")
    assert result == ["20240131", "20240229", "20240315"]


def test_collector_retries_transient_request_failure(monkeypatch):
    class FlakyClient:
        def __init__(self):
            self.calls = 0

        def daily(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise requests.Timeout("temporary timeout")
            return pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.0]})

    client = FlakyClient()
    collector = TushareSnapshotCollector(
        client,
        config=CollectionConfig(throttle_seconds=0, max_retries=1),
    )
    monkeypatch.setattr("research.collector.time.sleep", lambda _: None)

    result = collector._call("daily", trade_date="20260101")

    assert client.calls == 2
    assert result.iloc[0]["ts_code"] == "000001.SZ"


def test_collector_retries_empty_required_response(monkeypatch):
    class EmptyThenReadyClient:
        def __init__(self):
            self.calls = 0

        def daily(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return pd.DataFrame()
            return pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.0]})

    client = EmptyThenReadyClient()
    collector = TushareSnapshotCollector(
        client,
        config=CollectionConfig(throttle_seconds=0, max_retries=1),
    )
    monkeypatch.setattr("research.collector.time.sleep", lambda _: None)

    result = collector._call("daily", require_rows=True, trade_date="20260101")

    assert client.calls == 2
    assert result.iloc[0]["ts_code"] == "000001.SZ"


def test_storage_training_inference_and_backtest(tmp_path):
    store = _research_store(tmp_path)
    dates = pd.date_range("2023-01-31", periods=14, freq=pd.offsets.MonthEnd())
    for seed, value in enumerate(dates, start=1):
        store.replace_snapshot(_snapshot(value.strftime("%Y%m%d"), seed))

    status = store.status()
    assert status["snapshot_dates"] == 14
    assert status["snapshot_rows"] == 14 * 80

    metrics = train_and_register(
        store,
        TrainingConfig(min_train_dates=6, validation_dates=6, overlay_weight=0.15),
    )
    assert metrics["status"] == "approved"
    model = store.latest_model("approved")
    assert model is not None
    assert model.trained_through == dates[-1].strftime("%Y%m%d")

    latest = _snapshot(datetime.now().strftime("%Y%m%d"), 99)
    universe = latest.rename(columns={"industry": "sw_industry_name"})
    scored, _ = score_universe(universe, top_n=20)
    overlaid = apply_ml_overlay(scored, model_record=model)
    assert {"linear_score", "ml_score", "final_score", "final_rank"}.issubset(overlaid.columns)
    assert overlaid["final_score"].is_monotonic_decreasing
    assert overlaid["model_version"].nunique() == 1
    assert set(overlaid["model_status"]) == {"approved"}

    candidate = replace(model, status="candidate")
    with pytest.raises(RuntimeError, match="not approved"):
        apply_ml_overlay(scored, model_record=candidate)
    experimental = apply_ml_overlay(scored, model_record=candidate, allow_candidate=True)
    assert set(experimental["model_status"]) == {"candidate"}
    assert set(experimental["model_type"]) == {model.bundle["model_type"]}

    results, backtest_metrics = run_snapshot_backtest(store, top_n=20, transaction_cost=0.001)
    assert len(results) == 14
    assert np.isfinite(results["portfolio_nav"]).all()
    assert backtest_metrics["periods"] == 14
