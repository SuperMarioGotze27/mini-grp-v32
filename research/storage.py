"""Persistent research storage for snapshots and approved model artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    and_,
    create_engine,
    delete,
    func,
    insert,
    select,
)
from sqlalchemy.engine import Engine


FACTOR_COLUMNS = [
    "pe_ttm",
    "pb_lf",
    "ps_ttm",
    "dividend_yield",
    "gross_margin",
    "net_margin",
    "revenue_yoy",
    "profit_yoy",
    "return_1m",
    "return_3m",
    "return_12m",
]


metadata = MetaData()

snapshots = Table(
    "research_snapshots",
    metadata,
    Column("snapshot_date", String(8), primary_key=True),
    Column("code", String(20), primary_key=True),
    Column("label_date", String(8), nullable=True),
    Column("name", String(120), nullable=True),
    Column("industry", String(120), nullable=True),
    Column("market", String(8), nullable=False, default="CN"),
    Column("forward_return", Float, nullable=True),
    *[Column(column, Float, nullable=True) for column in FACTOR_COLUMNS],
    Column("created_at", DateTime(timezone=True), nullable=False),
)

models = Table(
    "model_registry",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(120), nullable=False),
    Column("version", String(80), nullable=False, unique=True),
    Column("status", String(20), nullable=False),
    Column("trained_from", String(8), nullable=False),
    Column("trained_through", String(8), nullable=False),
    Column("artifact", LargeBinary, nullable=False),
    Column("metrics_json", Text, nullable=False),
    Column("features_json", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)


def default_database_url() -> str:
    configured = os.environ.get("DATABASE_URL", "").strip()
    if configured:
        if configured.startswith("postgres://"):
            return configured.replace("postgres://", "postgresql+psycopg://", 1)
        if configured.startswith("postgresql://"):
            return configured.replace("postgresql://", "postgresql+psycopg://", 1)
        return configured
    path = Path(os.environ.get("MINI_GRP_DATA_DIR", "./research_data")).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(path / 'mini_grp_research.db').as_posix()}"


@dataclass
class ModelRecord:
    id: int
    name: str
    version: str
    status: str
    trained_from: str
    trained_through: str
    metrics: dict[str, Any]
    features: list[str]
    bundle: dict[str, Any]
    created_at: datetime


class ResearchStore:
    """SQL-backed snapshot store and model registry."""

    def __init__(self, database_url: Optional[str] = None) -> None:
        self.database_url = database_url or default_database_url()
        self.engine: Engine = create_engine(self.database_url, pool_pre_ping=True)
        metadata.create_all(self.engine)

    def replace_snapshot(self, frame: pd.DataFrame) -> int:
        if frame is None or frame.empty:
            raise ValueError("Cannot store an empty research snapshot")
        required = {"snapshot_date", "code"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Snapshot is missing required columns: {sorted(missing)}")
        clean = frame.copy()
        clean["snapshot_date"] = clean["snapshot_date"].astype(str)
        clean["code"] = clean["code"].astype(str)
        clean["market"] = clean.get("market", "CN")
        clean["created_at"] = datetime.now(timezone.utc)
        allowed = [column.name for column in snapshots.columns]
        for column in allowed:
            if column not in clean.columns:
                clean[column] = None
        clean = clean[allowed].where(pd.notna(clean), None)
        dates = sorted(clean["snapshot_date"].unique().tolist())
        with self.engine.begin() as connection:
            connection.execute(delete(snapshots).where(snapshots.c.snapshot_date.in_(dates)))
            connection.execute(insert(snapshots), clean.to_dict(orient="records"))
        return len(clean)

    def load_snapshots(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        labelled_only: bool = False,
    ) -> pd.DataFrame:
        query = select(snapshots)
        conditions = []
        if start_date:
            conditions.append(snapshots.c.snapshot_date >= start_date)
        if end_date:
            conditions.append(snapshots.c.snapshot_date <= end_date)
        if labelled_only:
            conditions.append(snapshots.c.forward_return.is_not(None))
        if conditions:
            query = query.where(and_(*conditions))
        query = query.order_by(snapshots.c.snapshot_date, snapshots.c.code)
        return pd.read_sql(query, self.engine)

    def status(self) -> dict[str, Any]:
        with self.engine.connect() as connection:
            snapshot_count = int(connection.execute(select(func.count()).select_from(snapshots)).scalar_one())
            date_count = int(
                connection.execute(select(func.count(func.distinct(snapshots.c.snapshot_date)))).scalar_one()
            )
            min_date, max_date = connection.execute(
                select(func.min(snapshots.c.snapshot_date), func.max(snapshots.c.snapshot_date))
            ).one()
            model_count = int(connection.execute(select(func.count()).select_from(models)).scalar_one())
        return {
            "snapshot_rows": snapshot_count,
            "snapshot_dates": date_count,
            "snapshot_start": min_date,
            "snapshot_end": max_date,
            "model_count": model_count,
            "database": self.database_url.split("@")[-1],
        }

    def save_model(
        self,
        bundle: dict[str, Any],
        metrics: dict[str, Any],
        features: list[str],
        status: str,
        name: str = "mini-grp-overlay",
    ) -> str:
        if status not in {"approved", "candidate", "rejected"}:
            raise ValueError("Model status must be approved, candidate, or rejected")
        trained_from = str(bundle["trained_from"])
        trained_through = str(bundle["trained_through"])
        version = f"{name}-{trained_through}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        buffer = BytesIO()
        joblib.dump(bundle, buffer)
        payload = {
            "name": name,
            "version": version,
            "status": status,
            "trained_from": trained_from,
            "trained_through": trained_through,
            "artifact": buffer.getvalue(),
            "metrics_json": json.dumps(metrics, ensure_ascii=False),
            "features_json": json.dumps(features, ensure_ascii=False),
            "created_at": datetime.now(timezone.utc),
        }
        with self.engine.begin() as connection:
            connection.execute(insert(models).values(**payload))
        return version

    def latest_model(self, status: str = "approved") -> Optional[ModelRecord]:
        query = (
            select(models)
            .where(models.c.status == status)
            .order_by(models.c.created_at.desc())
            .limit(1)
        )
        with self.engine.connect() as connection:
            row = connection.execute(query).mappings().first()
        if row is None:
            return None
        bundle = joblib.load(BytesIO(row["artifact"]))
        return ModelRecord(
            id=int(row["id"]),
            name=str(row["name"]),
            version=str(row["version"]),
            status=str(row["status"]),
            trained_from=str(row["trained_from"]),
            trained_through=str(row["trained_through"]),
            metrics=json.loads(row["metrics_json"]),
            features=json.loads(row["features_json"]),
            bundle=bundle,
            created_at=row["created_at"],
        )
