from datetime import date
from types import SimpleNamespace

import pandas as pd

from data.tushare_client import create_tushare_client, latest_open_trade_date, recent_open_trade_dates
from data.unified_fetcher import A_SHARE_FIELD_MAP, _fetch_tushare_financials, _map_columns


class FakeClient:
    def trade_cal(self, **kwargs):
        return pd.DataFrame(
            {
                "exchange": ["SSE", "SSE", "SSE"],
                "cal_date": ["20260610", "20260611", "20260612"],
                "is_open": [1, 1, 0],
            }
        )

    def fina_indicator(self, ts_code, **kwargs):
        return pd.DataFrame(
            {
                "ts_code": [ts_code, ts_code],
                "ann_date": ["20260430", "20260320"],
                "end_date": ["20260331", "20251231"],
                "roe_dt": [12.5, 11.0],
                "roa": [6.0, 5.5],
            }
        )


def test_map_columns_coalesces_tushare_aliases_without_duplicates():
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "600000.SH"],
            "pe": [9.0, 8.0],
            "pe_ttm": [8.5, None],
            "ps": [1.1, 1.2],
            "ps_ttm": [1.0, 1.15],
        }
    )
    mapped = _map_columns(raw, A_SHARE_FIELD_MAP)
    assert not mapped.columns.duplicated().any()
    assert mapped["pe_ttm"].tolist() == [8.5, 8.0]
    assert mapped["ps_ttm"].tolist() == [1.0, 1.15]


def test_latest_open_trade_date_uses_exchange_calendar():
    assert latest_open_trade_date(FakeClient(), as_of=date(2026, 6, 12)) == "20260611"
    assert recent_open_trade_dates(FakeClient(), as_of=date(2026, 6, 12)) == ["20260611", "20260610"]


def test_client_applies_custom_proxy_url(monkeypatch):
    client = SimpleNamespace()
    fake_module = SimpleNamespace(pro_api=lambda token: client)
    monkeypatch.setitem(__import__("sys").modules, "tushare", fake_module)
    result = create_tushare_client("secret", "https://example.test/api")
    assert result is client
    assert result._DataApi__http_url == "https://example.test/api"


def test_financial_enrichment_uses_full_dates_and_latest_disclosure():
    result = _fetch_tushare_financials(FakeClient(), ["000001.SZ"])
    assert len(result) == 1
    assert result.loc[0, "code"] == "000001.SZ"
    assert result.loc[0, "roa"] == 6.0
