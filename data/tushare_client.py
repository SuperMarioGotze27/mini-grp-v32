"""Shared Tushare client helpers for official and compatible proxy APIs."""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any, Optional
from urllib.parse import urlparse

import pandas as pd


class TushareClientError(RuntimeError):
    """Raised when a Tushare client cannot be configured or validated."""


def normalize_api_url(api_url: Optional[str]) -> str:
    """Validate and normalize an optional Tushare-compatible API URL."""
    value = (api_url or "").strip().rstrip("/")
    if not value:
        return ""
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise TushareClientError("TUSHARE_API_URL must be a valid HTTP(S) URL.")
    return value


def create_tushare_client(
    token: Optional[str] = None,
    api_url: Optional[str] = None,
    *,
    validate: bool = False,
) -> Any:
    """Create a Tushare client and optionally validate its trade calendar."""
    effective_token = (token or os.environ.get("TUSHARE_TOKEN", "")).strip()
    effective_url = normalize_api_url(api_url or os.environ.get("TUSHARE_API_URL", ""))
    if not effective_token:
        raise TushareClientError(
            "TUSHARE_TOKEN is not configured. Add it to Streamlit Secrets or enter it in the sidebar."
        )

    try:
        import tushare as ts
    except ImportError as exc:
        raise TushareClientError(
            "The tushare package is not installed. Install dependencies from requirements.txt."
        ) from exc

    try:
        client = ts.pro_api(effective_token)
        if effective_url:
            setattr(client, "_DataApi__http_url", effective_url)
        if validate:
            latest_open_trade_date(client)
        return client
    except Exception as exc:
        message = str(exc).replace(effective_token, "[REDACTED]")
        raise TushareClientError(f"Tushare connection failed: {message}") from exc


def recent_open_trade_dates(
    client: Any,
    as_of: Optional[date] = None,
    lookback_days: int = 20,
) -> list[str]:
    """Return recent SSE open dates in descending order."""
    end = as_of or date.today()
    start = end - timedelta(days=lookback_days)
    try:
        calendar = client.trade_cal(
            exchange="SSE",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            fields="exchange,cal_date,is_open",
        )
    except Exception as exc:
        raise TushareClientError(f"Unable to query the Tushare trade calendar: {exc}") from exc

    if calendar is None or calendar.empty:
        raise TushareClientError("Tushare returned an empty trade calendar.")
    open_days = calendar.loc[pd.to_numeric(calendar["is_open"], errors="coerce") == 1, "cal_date"]
    if open_days.empty:
        raise TushareClientError("No open trading day was found in the recent trade calendar.")
    return sorted(open_days.astype(str).unique().tolist(), reverse=True)


def latest_open_trade_date(client: Any, as_of: Optional[date] = None, lookback_days: int = 20) -> str:
    """Return the latest SSE open date on or before ``as_of`` in YYYYMMDD form."""
    return recent_open_trade_dates(client, as_of, lookback_days)[0]


def probe_tushare_connection(token: Optional[str] = None, api_url: Optional[str] = None) -> dict[str, Any]:
    """Validate credentials and return a small, non-sensitive connection summary."""
    client = create_tushare_client(token, api_url)
    trade_date = latest_open_trade_date(client)
    stocks = client.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,industry,list_date",
    )
    if stocks is None or stocks.empty:
        raise TushareClientError("Tushare authentication succeeded but stock_basic returned no rows.")
    return {"trade_date": trade_date, "listed_stocks": int(len(stocks))}
