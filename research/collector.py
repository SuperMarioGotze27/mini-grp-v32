"""Point-in-time monthly A-share snapshot collection from Tushare."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

from data.tushare_client import create_tushare_client
from research.storage import FACTOR_COLUMNS, ResearchStore

logger = logging.getLogger(__name__)


class EmptyTushareResponse(RuntimeError):
    """Raised when a required Tushare endpoint unexpectedly returns no rows."""


@dataclass(frozen=True)
class CollectionConfig:
    months: int = 48
    max_stocks: int = 1500
    forward_days: int = 20
    throttle_seconds: float = 0.42
    max_retries: int = 3
    end_date: Optional[str] = None


class TushareSnapshotCollector:
    """Collect monthly cross-sections and future-return labels without look-ahead."""

    def __init__(self, client: Any, config: CollectionConfig) -> None:
        self.client = client
        self.config = config
        self._last_call = 0.0

    def _call(self, method: str, require_rows: bool = False, **kwargs: Any) -> pd.DataFrame:
        for attempt in range(self.config.max_retries + 1):
            elapsed = time.monotonic() - self._last_call
            wait = self.config.throttle_seconds - elapsed
            if wait > 0:
                time.sleep(wait)
            try:
                result = getattr(self.client, method)(**kwargs)
                self._last_call = time.monotonic()
                frame = pd.DataFrame() if result is None else result
                if require_rows and frame.empty:
                    raise EmptyTushareResponse(f"{method} returned no rows")
                return frame
            except (requests.RequestException, EmptyTushareResponse) as exc:
                self._last_call = time.monotonic()
                if attempt >= self.config.max_retries:
                    raise
                retry_delay = min(2 ** attempt, 8)
                logger.warning(
                    "%s request failed (%d/%d): %s; retrying in %ss",
                    method,
                    attempt + 1,
                    self.config.max_retries + 1,
                    exc,
                    retry_delay,
                )
                time.sleep(retry_delay)
        raise RuntimeError(f"Unreachable retry state for Tushare method {method}")

    def open_dates(self) -> list[str]:
        snapshot_end = (
            datetime.strptime(self.config.end_date, "%Y%m%d").date()
            if self.config.end_date
            else date.today()
        )
        # Keep future trading dates in the calendar so historical cutoffs can
        # still receive their forward-return labels.
        calendar_end = max(snapshot_end, date.today())
        start = snapshot_end - timedelta(days=max(500, self.config.months * 35 + 420))
        calendar = self._call(
            "trade_cal",
            require_rows=True,
            exchange="SSE",
            start_date=start.strftime("%Y%m%d"),
            end_date=calendar_end.strftime("%Y%m%d"),
            fields="cal_date,is_open",
        )
        if calendar.empty:
            raise RuntimeError("Tushare returned an empty trade calendar")
        dates = calendar.loc[pd.to_numeric(calendar["is_open"], errors="coerce") == 1, "cal_date"]
        return sorted(dates.astype(str).unique().tolist())

    @staticmethod
    def month_end_dates(
        open_dates: list[str],
        months: int,
        end_date: Optional[str] = None,
    ) -> list[str]:
        frame = pd.DataFrame({"date": pd.to_datetime(open_dates)})
        if end_date:
            frame = frame[frame["date"] <= pd.to_datetime(end_date)]
        month_ends = frame.groupby(frame["date"].dt.to_period("M"))["date"].max().dt.strftime("%Y%m%d")
        return month_ends.tail(months).tolist()

    def _daily_close(self, trade_date: str) -> pd.DataFrame:
        frame = self._call(
            "daily",
            require_rows=True,
            trade_date=trade_date,
            fields="ts_code,trade_date,close",
        )
        return frame[["ts_code", "close"]].rename(columns={"ts_code": "code"})

    def collect_date(self, snapshot_date: str, open_dates: list[str]) -> pd.DataFrame:
        index = open_dates.index(snapshot_date)
        daily_basic = self._call(
            "daily_basic",
            require_rows=True,
            trade_date=snapshot_date,
            fields=(
                "ts_code,trade_date,close,pe_ttm,pb,ps_ttm,dv_ttm,"
                "total_mv,circ_mv"
            ),
        )
        if daily_basic.empty:
            raise RuntimeError(f"daily_basic returned no rows for {snapshot_date}")
        base = daily_basic.rename(
            columns={
                "ts_code": "code",
                "pb": "pb_lf",
                "dv_ttm": "dividend_yield",
            }
        )

        try:
            quality = self._call(
                "bak_basic",
                trade_date=snapshot_date,
                fields="trade_date,ts_code,name,industry,rev_yoy,profit_yoy,gpr,npr",
            ).rename(
                columns={
                    "ts_code": "code",
                    "industry": "industry",
                    "rev_yoy": "revenue_yoy",
                    "gpr": "gross_margin",
                    "npr": "net_margin",
                }
            )
        except Exception as exc:
            logger.warning("bak_basic unavailable for %s: %s", snapshot_date, exc)
            quality = pd.DataFrame()
        if not quality.empty:
            base = base.merge(quality, on="code", how="left", suffixes=("", "_quality"))

        horizons = {"1m": 22, "3m": 66, "12m": 252}
        for label, offset in horizons.items():
            if index < offset:
                continue
            past = self._daily_close(open_dates[index - offset]).rename(columns={"close": f"close_{label}"})
            base = base.merge(past, on="code", how="left")
            base[f"return_{label}"] = (
                pd.to_numeric(base["close"], errors="coerce")
                / pd.to_numeric(base[f"close_{label}"], errors="coerce")
                - 1
            ) * 100

        label_index = index + self.config.forward_days
        label_date = open_dates[label_index] if label_index < len(open_dates) else None
        if label_date:
            future = self._daily_close(label_date).rename(columns={"close": "future_close"})
            base = base.merge(future, on="code", how="left")
            base["forward_return"] = (
                pd.to_numeric(base["future_close"], errors="coerce")
                / pd.to_numeric(base["close"], errors="coerce")
                - 1
            )
        else:
            base["forward_return"] = np.nan

        base["snapshot_date"] = snapshot_date
        base["label_date"] = label_date
        base["market"] = "CN"
        if "industry" not in base:
            base["industry"] = "Unknown"
        if "name" not in base:
            base["name"] = base["code"]
        if "total_mv" in base:
            base = base.sort_values("total_mv", ascending=False)
        base = base.head(self.config.max_stocks).copy()
        keep = [
            "snapshot_date",
            "label_date",
            "code",
            "name",
            "industry",
            "market",
            "forward_return",
            *FACTOR_COLUMNS,
        ]
        for column in keep:
            if column not in base:
                base[column] = np.nan
        return base[keep].replace([np.inf, -np.inf], np.nan)

    def collect(self, store: ResearchStore) -> dict[str, Any]:
        open_dates = self.open_dates()
        month_ends = self.month_end_dates(open_dates, self.config.months, self.config.end_date)
        existing = store.load_snapshots()
        existing_dates = set(existing["snapshot_date"].astype(str)) if not existing.empty else set()
        existing_counts = (
            existing.groupby("snapshot_date")["code"].nunique().to_dict()
            if not existing.empty
            else {}
        )
        refresh_dates = set(month_ends[-2:])
        pending = [
            value
            for value in month_ends
            if value not in existing_dates
            or value in refresh_dates
            or int(existing_counts.get(value, 0)) < self.config.max_stocks
        ]
        rows = 0
        completed = []
        for position, snapshot_date in enumerate(pending, start=1):
            logger.info("Collecting snapshot %s (%d/%d)", snapshot_date, position, len(pending))
            frame = self.collect_date(snapshot_date, open_dates)
            rows += store.replace_snapshot(frame)
            completed.append(snapshot_date)
        return {
            "dates": completed,
            "rows": rows,
            "skipped_existing_dates": len(month_ends) - len(pending),
        }


def collect_history(config: CollectionConfig, store: ResearchStore) -> dict[str, Any]:
    client = create_tushare_client(validate=True)
    return TushareSnapshotCollector(client, config).collect(store)
