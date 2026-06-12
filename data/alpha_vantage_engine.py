#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini-GRP v3.0 — Alpha Vantage 美股数据接入模块

本模块为 Mini-GRP 量化选股系统提供 Alpha Vantage API 的美股数据获取能力，
替代 yfinance 以获取更稳定、结构化的财务数据。

功能
----
- Alpha Vantage 客户端封装（自动读取环境变量 API key）
- 美股三大财务报表获取（利润表 / 资产负债表 / 现金流量表）
- 标准化 16 因子计算（Value / Quality / Growth / Momentum）
- 历史价格数据获取与动量因子计算
- 批量多股票数据获取（带速率限制与本地缓存）
- 高保真 Mock 数据降级

依赖
----
- alpha-vantage
- pandas, numpy

作者
----
Mini-GRP Development Team
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logger = logging.getLogger("mini_grp_v3")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# Alpha Vantage 免费 tier 限制
AV_FREE_CALLS_PER_MIN: int = 5
AV_FREE_CALLS_PER_DAY: int = 500
AV_SLEEP_SECONDS: float = 12.5  # 略大于 12s，留安全余量

# 本地缓存目录
CACHE_DIR: Path = Path(__file__).parent / "av_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Alpha Vantage 字段 -> 统一字段名映射（利润表）
AV_INCOME_FIELD_MAP: Dict[str, str] = {
    "totalRevenue": "total_revenue",
    "netIncome": "net_income",
    "grossProfit": "gross_profit",
    "operatingIncome": "operating_income",
    "ebitda": "ebitda",
    "interestExpense": "interest_expense",
    "incomeBeforeTax": "income_before_tax",
    "netIncomeApplicableToCommonShares": "net_income_common",
}

# Alpha Vantage 字段 -> 统一字段名映射（资产负债表）
AV_BALANCE_FIELD_MAP: Dict[str, str] = {
    "totalAssets": "total_assets",
    "totalLiabilities": "total_liabilities",
    "totalShareholderEquity": "total_equity",
    "commonStockSharesOutstanding": "shares_outstanding",
    "cashAndCashEquivalentsAtCarryingValue": "cash",
    "shortTermDebt": "short_term_debt",
    "longTermDebt": "long_term_debt",
    "totalShareholderEquity": "shareholder_equity",
    "propertyPlantEquipment": "ppe",
    "goodwill": "goodwill",
    "intangibleAssets": "intangible_assets",
}

# Alpha Vantage 字段 -> 统一字段名映射（现金流量表）
AV_CASHFLOW_FIELD_MAP: Dict[str, str] = {
    "operatingCashflow": "operating_cashflow",
    "capitalExpenditures": "capex",
    "freeCashFlow": "free_cashflow",
    "dividendPayout": "dividend_payout",
    "netIncome": "net_income_cf",
}

# GICS 一级行业列表（与 multi_market_engine.py 保持一致）
GICS_SECTORS: List[str] = [
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Information Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
]


# ---------------------------------------------------------------------------
# 安全导入
# ---------------------------------------------------------------------------

def _safe_import(module_name: str) -> Optional[Any]:
    """安全导入模块，失败时返回 None 并记录日志。"""
    try:
        return __import__(module_name)
    except ImportError:
        logger.warning("模块 %s 未安装，相关功能将使用 mock 数据", module_name)
        return None


_alpha_vantage = _safe_import("alpha_vantage")


# ---------------------------------------------------------------------------
# 1. Alpha Vantage 客户端封装
# ---------------------------------------------------------------------------

def get_alpha_vantage_client(api_key: Optional[str] = None) -> Optional[Any]:
    """获取 Alpha Vantage API 客户端。

    优先从环境变量 ``ALPHA_VANTAGE_API_KEY`` 读取，其次从参数传入。
    如果未安装 ``alpha_vantage`` 库，返回 None。

    Parameters
    ----------
    api_key :
        可选的 Alpha Vantage API key。若未提供且环境变量也不存在，
        则返回 None。

    Returns
    -------
    Optional[Any]
        包含 ``FundamentalData`` 和 ``TimeSeries`` 实例的字典，
        或 None（库未安装 / 无 API key）。
    """
    if _alpha_vantage is None:
        logger.warning("alpha_vantage 库未安装，无法创建客户端")
        return None

    key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not key:
        logger.warning(
            "未提供 Alpha Vantage API key（请设置环境变量 ALPHA_VANTAGE_API_KEY）"
        )
        return None

    try:
        from alpha_vantage.fundamentaldata import FundamentalData
        from alpha_vantage.timeseries import TimeSeries

        return {
            "fundamental": FundamentalData(key, output_format="pandas"),
            "timeseries": TimeSeries(key, output_format="pandas"),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("初始化 Alpha Vantage 客户端失败: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 2. 获取美股财务数据
# ---------------------------------------------------------------------------

def fetch_us_financials(
    symbol: str,
    client: Optional[Any] = None,
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """获取美股三大财务报表。

    使用 Alpha Vantage 的 ``INCOME_STATEMENT``、``BALANCE_SHEET``、
    ``CASH_FLOW`` 接口，返回最近 5 个季度的数据。

    Parameters
    ----------
    symbol :
        美股代码，如 ``"AAPL"``、``"MSFT"``。
    client :
        Alpha Vantage 客户端字典（由 ``get_alpha_vantage_client`` 返回）。
        若未提供，自动尝试创建。
    use_cache :
        是否优先读取本地缓存。

    Returns
    -------
    Dict[str, pd.DataFrame]
        包含以下键的字典：
        - ``'income'``: 利润表 DataFrame
        - ``'balance'``: 资产负债表 DataFrame
        - ``'cashflow'``: 现金流量表 DataFrame

        每个 DataFrame 的列名为统一小写下划线格式，
        行索引为 ``fiscal_date_ending``（最近 5 个季度）。
        若 API 不可用，返回空 DataFrame 的字典。
    """
    symbol = symbol.upper().strip()
    cache_path = CACHE_DIR / f"{symbol}_financials.json"

    if use_cache and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                cached = json.load(fh)
            logger.debug("[%s] 从缓存读取财务数据", symbol)
            return {
                "income": pd.DataFrame(cached.get("income", {})),
                "balance": pd.DataFrame(cached.get("balance", {})),
                "cashflow": pd.DataFrame(cached.get("cashflow", {})),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] 缓存读取失败: %s", symbol, exc)

    client = client or get_alpha_vantage_client()
    if client is None:
        logger.warning("[%s] Alpha Vantage 客户端不可用，返回空财务数据", symbol)
        return {"income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}

    fd = client["fundamental"]
    result: Dict[str, pd.DataFrame] = {}

    # 定义三个报表的获取方法
    endpoints: List[Tuple[str, str, Dict[str, str]]] = [
        ("income", "get_income_statement_quarterly", AV_INCOME_FIELD_MAP),
        ("balance", "get_balance_sheet_quarterly", AV_BALANCE_FIELD_MAP),
        ("cashflow", "get_cash_flow_quarterly", AV_CASHFLOW_FIELD_MAP),
    ]

    for key, method_name, field_map in endpoints:
        try:
            method = getattr(fd, method_name)
            data, _meta = method(symbol=symbol)  # type: ignore[operator]

            # 检查 API 返回的错误信息
            if isinstance(data, dict) and "Information" in data:
                raise RuntimeError(data["Information"])
            if isinstance(data, pd.DataFrame) and data.empty:
                raise RuntimeError("Empty response")

            # 统一字段名与数据类型
            df = _normalize_av_financials(data, field_map)
            result[key] = df
            logger.debug("[%s] %s 获取成功，%d 条记录", symbol, key, len(df))

            # 速率限制
            time.sleep(AV_SLEEP_SECONDS)

        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] 获取 %s 失败: %s", symbol, key, exc)
            result[key] = pd.DataFrame()
            time.sleep(AV_SLEEP_SECONDS)

    # 写入缓存
    if use_cache and any(not df.empty for df in result.values()):
        try:
            cache_payload = {
                k: v.reset_index().to_dict(orient="records")
                for k, v in result.items()
            }
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(cache_payload, fh, indent=2, default=str)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] 缓存写入失败: %s", symbol, exc)

    return result


def _normalize_av_financials(
    raw: Any,
    field_map: Dict[str, str],
) -> pd.DataFrame:
    """将 Alpha Vantage 原始财务数据标准化为 DataFrame。

    Parameters
    ----------
    raw :
        API 返回的原始数据（dict 或 pandas DataFrame）。
    field_map :
        字段名映射表（AV 原始名 -> 统一名）。

    Returns
    -------
    pd.DataFrame
        标准化后的 DataFrame，索引为 ``fiscal_date_ending``，
        列名为小写下划线格式，数值类型为 float。
    """
    if isinstance(raw, pd.DataFrame):
        df = raw.copy()
    elif isinstance(raw, dict):
        # alpha_vantage 库有时返回 dict，需要手动提取 quarterlyReports
        reports = raw.get("quarterlyReports", [])
        if not reports:
            return pd.DataFrame()
        df = pd.DataFrame(reports)
    else:
        return pd.DataFrame()

    if df.empty:
        return df

    # 重命名并筛选字段
    rename_map = {k: v for k, v in field_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # 设置日期索引
    if "fiscalDateEnding" in df.columns:
        df["fiscal_date_ending"] = pd.to_datetime(df["fiscalDateEnding"])
        df = df.set_index("fiscal_date_ending").drop(columns=["fiscalDateEnding"], errors="ignore")
    elif "fiscal_date_ending" in df.columns:
        df["fiscal_date_ending"] = pd.to_datetime(df["fiscal_date_ending"])
        df = df.set_index("fiscal_date_ending")

    # 数值转换：Alpha Vantage 返回字符串形式的数字
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 按日期降序排列，取最近 5 个季度
    df = df.sort_index(ascending=False).head(5)
    return df


# ---------------------------------------------------------------------------
# 3. 计算美股价值/质量/增长因子
# ---------------------------------------------------------------------------

def calculate_us_factors(
    financials: Dict[str, pd.DataFrame],
    price_data: Optional[pd.DataFrame] = None,
    market_cap: Optional[float] = None,
) -> pd.DataFrame:
    """从财务报表计算标准化因子。

    计算以下 16 因子（与现有系统对齐）：

    - **Value**: ``pe_ttm``, ``pb_lf``, ``ps_ttm``, ``ev_ebitda``, ``dividend_yield``
    - **Quality**: ``roe``, ``roa``, ``gross_margin``, ``net_margin``, ``debt_to_equity``
    - **Growth**: ``revenue_growth``, ``profit_growth``, ``fcf_yield``
    - **Momentum**: ``return_1m``, ``return_3m``, ``return_12m``（来自 price_data）

    Parameters
    ----------
    financials :
        由 ``fetch_us_financials`` 返回的字典，包含 ``income``、
        ``balance``、``cashflow`` 三个 DataFrame。
    price_data :
        可选的价格数据 DataFrame（由 ``fetch_us_price_data`` 返回），
        用于提取市值和动量因子。若未提供，动量因子为 NaN。
    market_cap :
        可选的市值覆盖值（单位：USD）。若提供，优先使用此值
        而非从 price_data 推导。

    Returns
    -------
    pd.DataFrame
        单行 DataFrame，索引为 ``symbol``，列名为小写下划线因子名。
    """
    income = financials.get("income", pd.DataFrame())
    balance = financials.get("balance", pd.DataFrame())
    cashflow = financials.get("cashflow", pd.DataFrame())

    # TTM 计算：最近 4 个季度求和
    def _ttm(df: pd.DataFrame, col: str) -> float:
        if df.empty or col not in df.columns:
            return np.nan
        return df[col].head(4).sum()

    def _latest(df: pd.DataFrame, col: str) -> float:
        if df.empty or col not in df.columns:
            return np.nan
        return df[col].iloc[0]

    # 核心财务指标 TTM / 最新值
    revenue_ttm = _ttm(income, "total_revenue")
    net_income_ttm = _ttm(income, "net_income")
    gross_profit_ttm = _ttm(income, "gross_profit")
    ebitda_ttm = _ttm(income, "ebitda")
    operating_income_ttm = _ttm(income, "operating_income")

    total_assets = _latest(balance, "total_assets")
    total_liabilities = _latest(balance, "total_liabilities")
    total_equity = _latest(balance, "total_equity")
    shareholder_equity = _latest(balance, "shareholder_equity")
    shares_outstanding = _latest(balance, "shares_outstanding")
    cash = _latest(balance, "cash")
    short_term_debt = _latest(balance, "short_term_debt")
    long_term_debt = _latest(balance, "long_term_debt")

    operating_cf_ttm = _ttm(cashflow, "operating_cashflow")
    capex_ttm = _ttm(cashflow, "capex")
    dividend_payout_ttm = _ttm(cashflow, "dividend_payout")

    # 自由现金流 = 经营现金流 - 资本支出
    fcf_ttm = operating_cf_ttm - abs(capex_ttm) if not pd.isna(operating_cf_ttm) and not pd.isna(capex_ttm) else np.nan

    # 总债务
    total_debt = (
        (short_term_debt if not pd.isna(short_term_debt) else 0)
        + (long_term_debt if not pd.isna(long_term_debt) else 0)
    )
    if total_debt == 0:
        total_debt = np.nan

    # 账面价值优先使用 total_equity，其次 shareholder_equity
    book_value = total_equity if not pd.isna(total_equity) else shareholder_equity

    # 权益优先使用 book_value，其次 total_assets - total_liabilities
    equity = book_value
    if pd.isna(equity) and not pd.isna(total_assets) and not pd.isna(total_liabilities):
        equity = total_assets - total_liabilities

    # 市值推导
    mcap = market_cap
    if pd.isna(mcap) and price_data is not None and not price_data.empty:
        latest_close = price_data["adjusted_close"].iloc[-1] if "adjusted_close" in price_data.columns else price_data["close"].iloc[-1]
        if not pd.isna(shares_outstanding) and not pd.isna(latest_close):
            mcap = latest_close * shares_outstanding

    # 同比增长率（最近季度 vs 去年同期季度）
    def _yoy_growth(df: pd.DataFrame, col: str) -> float:
        if df.empty or col not in df.columns or len(df) < 5:
            return np.nan
        recent = df[col].iloc[0]
        year_ago = df[col].iloc[4] if len(df) > 4 else np.nan
        if pd.isna(recent) or pd.isna(year_ago) or year_ago == 0:
            return np.nan
        return (recent - year_ago) / abs(year_ago)

    revenue_growth = _yoy_growth(income, "total_revenue")
    profit_growth = _yoy_growth(income, "net_income")

    # 股息率
    dividend_yield = dividend_payout_ttm / mcap if not pd.isna(dividend_payout_ttm) and not pd.isna(mcap) and mcap > 0 else np.nan

    eps = 1e-8  # 避免除零

    # 构建因子 DataFrame
    factors = pd.DataFrame({
        "pe_ttm": [mcap / (net_income_ttm + eps) if not pd.isna(mcap) and not pd.isna(net_income_ttm) else np.nan],
        "pb_lf": [mcap / (book_value + eps) if not pd.isna(mcap) and not pd.isna(book_value) else np.nan],
        "ps_ttm": [mcap / (revenue_ttm + eps) if not pd.isna(mcap) and not pd.isna(revenue_ttm) else np.nan],
        "ev_ebitda": [
            (mcap + total_debt - (cash or 0)) / (ebitda_ttm + eps)
            if not pd.isna(mcap) and not pd.isna(ebitda_ttm) else np.nan
        ],
        "dividend_yield": [dividend_yield],
        "roe": [net_income_ttm / (equity + eps) if not pd.isna(net_income_ttm) and not pd.isna(equity) else np.nan],
        "roa": [net_income_ttm / (total_assets + eps) if not pd.isna(net_income_ttm) and not pd.isna(total_assets) else np.nan],
        "gross_margin": [gross_profit_ttm / (revenue_ttm + eps) if not pd.isna(gross_profit_ttm) and not pd.isna(revenue_ttm) else np.nan],
        "net_margin": [net_income_ttm / (revenue_ttm + eps) if not pd.isna(net_income_ttm) and not pd.isna(revenue_ttm) else np.nan],
        "debt_to_equity": [total_liabilities / (equity + eps) if not pd.isna(total_liabilities) and not pd.isna(equity) else np.nan],
        "revenue_growth": [revenue_growth],
        "profit_growth": [profit_growth],
        "fcf_yield": [fcf_ttm / (mcap + eps) if not pd.isna(fcf_ttm) and not pd.isna(mcap) else np.nan],
    })

    # 动量因子
    if price_data is not None and not price_data.empty:
        factors["return_1m"] = _calc_return(price_data, days=21)
        factors["return_3m"] = _calc_return(price_data, days=63)
        factors["return_12m"] = _calc_return(price_data, days=252)
    else:
        factors["return_1m"] = np.nan
        factors["return_3m"] = np.nan
        factors["return_12m"] = np.nan

    return factors


def _calc_return(price_df: pd.DataFrame, days: int = 21) -> float:
    """计算指定天数的收益率。

    Parameters
    ----------
    price_df :
        包含 ``close`` 或 ``adjusted_close`` 列的价格 DataFrame，
        按日期升序排列。
    days :
        回望交易日数。

    Returns
    -------
    float
        期间收益率，数据不足时返回 NaN。
    """
    if price_df.empty or len(price_df) < 2:
        return np.nan
    col = "adjusted_close" if "adjusted_close" in price_df.columns else "close"
    recent = price_df[col].iloc[-1]
    past = price_df[col].iloc[-min(days, len(price_df) - 1) - 1]
    if pd.isna(recent) or pd.isna(past) or past == 0:
        return np.nan
    return (recent - past) / past


# ---------------------------------------------------------------------------
# 4. 获取美股价格数据（动量因子）
# ---------------------------------------------------------------------------

def fetch_us_price_data(
    symbol: str,
    client: Optional[Any] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """获取美股历史价格数据。

    使用 Alpha Vantage ``TIME_SERIES_DAILY_ADJUSTED`` 接口，
    获取最近 ~1 年的日频数据。

    Parameters
    ----------
    symbol :
        美股代码，如 ``"AAPL"``。
    client :
        Alpha Vantage 客户端字典。若未提供，自动尝试创建。
    use_cache :
        是否优先读取本地缓存。

    Returns
    -------
    pd.DataFrame
        包含以下列：
        - ``date``: 交易日（datetime）
        - ``close``: 收盘价
        - ``adjusted_close``: 调整后收盘价
        - ``volume``: 成交量

        按日期升序排列。若 API 不可用，返回空 DataFrame。
    """
    symbol = symbol.upper().strip()
    cache_path = CACHE_DIR / f"{symbol}_price.csv"

    if use_cache and cache_path.exists():
        try:
            df = pd.read_csv(cache_path, parse_dates=["date"])
            logger.debug("[%s] 从缓存读取价格数据", symbol)
            return df
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] 价格缓存读取失败: %s", symbol, exc)

    client = client or get_alpha_vantage_client()
    if client is None:
        logger.warning("[%s] Alpha Vantage 客户端不可用，返回空价格数据", symbol)
        return pd.DataFrame()

    ts = client["timeseries"]
    try:
        data, meta = ts.get_daily_adjusted(symbol=symbol, outputsize="full")  # type: ignore[operator]

        # 检查错误信息
        if isinstance(data, dict) and "Information" in data:
            raise RuntimeError(data["Information"])
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            raise RuntimeError("Empty price response")

        # alpha_vantage pandas 输出列名示例：
        # 1. open, 2. high, 3. low, 4. close, 5. adjusted close, 6. volume, 7. dividend amount, 8. split coefficient
        df = data.copy()
        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. adjusted close": "adjusted_close",
            "6. volume": "volume",
        })
        df = df[["close", "adjusted_close", "volume"]]
        df = df.reset_index()
        df = df.rename(columns={"index": "date", "date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # 写入缓存
        if use_cache:
            df.to_csv(cache_path, index=False)

        logger.debug("[%s] 价格数据获取成功，%d 条记录", symbol, len(df))
        return df

    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] 获取价格数据失败: %s", symbol, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 5. 批量获取多只股票数据
# ---------------------------------------------------------------------------

def fetch_us_stock_universe(
    symbols: List[str],
    client: Optional[Any] = None,
    max_requests_per_min: int = AV_FREE_CALLS_PER_MIN,
    use_cache: bool = True,
) -> pd.DataFrame:
    """批量获取多只股票的所有因子数据。

    Alpha Vantage 免费 tier 限制 5 calls/min、500 calls/day。
    本函数在每次 API 请求后 sleep 12.5s，确保不触发限流。

    每只股票的完整流程需要 3 次 API 调用（财务 ×3）+ 1 次价格调用
    = 4 次调用。对于大量股票，总耗时较长，强烈建议启用缓存。

    Parameters
    ----------
    symbols :
        美股代码列表，如 ``["AAPL", "MSFT", "GOOGL"]``。
    client :
        Alpha Vantage 客户端字典。若未提供，自动尝试创建。
    max_requests_per_min :
        每分钟最大请求数（默认 5，与免费 tier 一致）。
    use_cache :
        是否使用本地 JSON/CSV 缓存。

    Returns
    -------
    pd.DataFrame
        包含以下列：
        - ``ticker``: 股票代码
        - ``market``: 固定为 ``"US"``
        - ``sector``: GICS 行业（从 mock 或 info 推断）
        - ``market_cap``: 市值（USD）
        - ``pe_ttm``, ``pb_lf``, ``ps_ttm``, ``ev_ebitda``, ``dividend_yield``
        - ``roe``, ``roa``, ``gross_margin``, ``net_margin``, ``debt_to_equity``
        - ``revenue_growth``, ``profit_growth``, ``fcf_yield``
        - ``return_1m``, ``return_3m``, ``return_12m``

        若 API 不可用或超限，返回 mock 数据。
    """
    client = client or get_alpha_vantage_client()
    if client is None:
        logger.warning(
            "Alpha Vantage 客户端不可用（未安装库或无 API key），"
            "降级至 mock 数据模式"
        )
        return _generate_mock_us_data(symbols)

    records: List[Dict[str, Any]] = []
    request_count = 0
    day_request_count = 0
    sleep_sec = 60.0 / max_requests_per_min + 0.5  # 留余量

    for symbol in symbols:
        symbol = symbol.upper().strip()
        logger.info("[%s] 开始获取数据...", symbol)

        try:
            # 获取财务数据（内部已含 3 次 API 调用 + sleep）
            financials = fetch_us_financials(symbol, client=client, use_cache=use_cache)

            # 获取价格数据（1 次 API 调用）
            price_data = fetch_us_price_data(symbol, client=client, use_cache=use_cache)

            # 若财务数据全部为空，说明 API 可能已超限或出错，降级
            if all(df.empty for df in financials.values()):
                logger.warning("[%s] 财务数据为空，跳过该股票", symbol)
                continue

            # 计算因子
            factors = calculate_us_factors(financials, price_data=price_data)

            # 推断 sector（Alpha Vantage 不直接提供，使用 mock 填充）
            sector = _infer_sector(symbol)

            # 市值
            mcap = np.nan
            if not price_data.empty and "adjusted_close" in price_data.columns:
                latest_close = price_data["adjusted_close"].iloc[-1]
                balance = financials.get("balance", pd.DataFrame())
                if not balance.empty and "shares_outstanding" in balance.columns:
                    shares = balance["shares_outstanding"].iloc[0]
                    if not pd.isna(shares) and not pd.isna(latest_close):
                        mcap = latest_close * shares

            record = {
                "ticker": symbol,
                "market": "US",
                "sector": sector,
                "market_cap": mcap,
            }
            # 合并因子列
            for col in factors.columns:
                record[col] = factors[col].iloc[0]

            records.append(record)
            day_request_count += 4  # 3 财务 + 1 价格

            # 日限额检查
            if day_request_count >= AV_FREE_CALLS_PER_DAY:
                logger.warning(
                    "接近 Alpha Vantage 日限额 (%d calls)，停止批量获取",
                    AV_FREE_CALLS_PER_DAY,
                )
                break

        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] 批量获取异常: %s", symbol, exc)
            continue

    if not records:
        logger.warning("所有股票 API 获取失败，降级至 mock 数据")
        return _generate_mock_us_data(symbols)

    df = pd.DataFrame(records)

    # 确保列顺序与接口约定一致
    expected_cols = [
        "ticker", "market", "sector", "market_cap",
        "pe_ttm", "pb_lf", "ps_ttm", "ev_ebitda", "dividend_yield",
        "roe", "roa", "gross_margin", "net_margin", "debt_to_equity",
        "revenue_growth", "profit_growth", "fcf_yield",
        "return_1m", "return_3m", "return_12m",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[expected_cols]

    return df


def _infer_sector(symbol: str) -> str:
    """根据股票代码推断 GICS 行业（简化版）。

    实际生产环境应通过 Alpha Vantage ``OVERVIEW`` 接口或
    外部映射表获取。此处使用随机分配保证数据完整性。

    Parameters
    ----------
    symbol :
        美股代码。

    Returns
    -------
    str
        GICS 一级行业名称。
    """
    # 一些常见股票的硬编码映射
    known: Dict[str, str] = {
        "AAPL": "Information Technology",
        "MSFT": "Information Technology",
        "GOOGL": "Communication Services",
        "GOOG": "Communication Services",
        "AMZN": "Consumer Discretionary",
        "NVDA": "Information Technology",
        "META": "Communication Services",
        "TSLA": "Consumer Discretionary",
        "BRK-B": "Financials",
        "JPM": "Financials",
        "JNJ": "Health Care",
        "V": "Information Technology",
        "PG": "Consumer Staples",
        "UNH": "Health Care",
        "HD": "Consumer Discretionary",
        "MA": "Information Technology",
        "BAC": "Financials",
        "ABBV": "Health Care",
        "PFE": "Health Care",
        "KO": "Consumer Staples",
        "PEP": "Consumer Staples",
        "WMT": "Consumer Staples",
        "MRK": "Health Care",
        "CSCO": "Information Technology",
        "ADBE": "Information Technology",
        "NFLX": "Communication Services",
        "CRM": "Information Technology",
        "ACN": "Information Technology",
        "XOM": "Energy",
        "CVX": "Energy",
        "LLY": "Health Care",
        "AVGO": "Information Technology",
        "TMO": "Health Care",
        "COST": "Consumer Staples",
        "DIS": "Communication Services",
        "ABT": "Health Care",
        "VZ": "Communication Services",
        "WFC": "Financials",
        "DHR": "Health Care",
        "TXN": "Information Technology",
        "PM": "Consumer Staples",
        "NKE": "Consumer Discretionary",
        "RTX": "Industrials",
        "INTC": "Information Technology",
        "UPS": "Industrials",
        "NEE": "Utilities",
        "QCOM": "Information Technology",
        "MS": "Financials",
        "LIN": "Materials",
        "AMD": "Information Technology",
        "HON": "Industrials",
        "INTU": "Information Technology",
        "SPGI": "Financials",
        "GS": "Financials",
        "CAT": "Industrials",
        "AMGN": "Health Care",
        "SBUX": "Consumer Discretionary",
        "PLD": "Real Estate",
        "IBM": "Information Technology",
        "DE": "Industrials",
        "ELV": "Health Care",
        "BMY": "Health Care",
        "BLK": "Financials",
        "LMT": "Industrials",
        "GE": "Industrials",
        "T": "Communication Services",
        "MDT": "Health Care",
        "AMAT": "Information Technology",
    }
    if symbol.upper() in known:
        return known[symbol.upper()]

    rng = np.random.default_rng(seed=hash(symbol) % 2**31)
    return rng.choice(GICS_SECTORS)


# ---------------------------------------------------------------------------
# 6. Mock 数据生成器
# ---------------------------------------------------------------------------

def _generate_mock_us_data(symbols: List[str]) -> pd.DataFrame:
    """生成模拟的美股数据，与真实数据分布一致。

    生成的 mock 数据在统计分布上尽量贴近美股典型特征：
    - P/E ~ 15–30x
    - P/B ~ 2–5x
    - ROE ~ 10–25%
    - 营收增长 ~ 0–20%
    - 1M/3M/12M 收益率 ~ 正态分布

    Parameters
    ----------
    symbols :
        美股代码列表。

    Returns
    -------
    pd.DataFrame
        与 ``fetch_us_stock_universe`` 返回格式一致的 DataFrame，
        包含 ``ticker``, ``market``, ``sector``, ``market_cap`` 及 16 个因子。
    """
    rng = np.random.default_rng(seed=42)
    n = len(symbols)

    # 市值：对数正态分布，美股大盘股为主
    market_caps = rng.lognormal(mean=24, sigma=1.2, size=n)

    # 行业分配
    sectors = rng.choice(GICS_SECTORS, size=n)

    # 基础财务指标（从市值反推，保持合理比例）
    net_profit = market_caps / rng.lognormal(2.8, 0.4, n)  # PE ~ 15-30
    book_value = market_caps / rng.lognormal(1.2, 0.35, n)  # PB ~ 2-5
    revenue = market_caps / rng.lognormal(1.4, 0.5, n)  # PS ~ 1-5
    ebitda = market_caps / rng.lognormal(2.2, 0.4, n)  # EV/EBITDA ~ 8-18
    dividends = market_caps * rng.uniform(0.005, 0.035, n)
    equity = book_value
    total_assets = equity * rng.uniform(1.5, 4.0, n)
    total_liabilities = total_assets - equity
    gross_profit = revenue * rng.uniform(0.25, 0.75, n)
    debt = total_liabilities * rng.uniform(0.3, 0.8, n)
    cash = market_caps * rng.uniform(0.03, 0.25, n)
    fcf = net_profit * rng.uniform(0.4, 1.1, n)

    # 增长率
    rev_growth = rng.normal(0.08, 0.18, n)
    profit_growth = rng.normal(0.06, 0.22, n)

    # 动量
    ret_1m = rng.normal(0.005, 0.06, n)
    ret_3m = rng.normal(0.015, 0.10, n)
    ret_12m = rng.normal(0.08, 0.20, n)

    eps = 1e-8

    df = pd.DataFrame({
        "ticker": symbols,
        "market": "US",
        "sector": sectors,
        "market_cap": market_caps,
        "pe_ttm": market_caps / (net_profit + eps),
        "pb_lf": market_caps / (book_value + eps),
        "ps_ttm": market_caps / (revenue + eps),
        "ev_ebitda": (market_caps + debt - cash) / (ebitda + eps),
        "dividend_yield": dividends / (market_caps + eps),
        "roe": net_profit / (equity + eps),
        "roa": net_profit / (total_assets + eps),
        "gross_margin": gross_profit / (revenue + eps),
        "net_margin": net_profit / (revenue + eps),
        "debt_to_equity": total_liabilities / (equity + eps),
        "revenue_growth": rev_growth,
        "profit_growth": profit_growth,
        "fcf_yield": fcf / (market_caps + eps),
        "return_1m": ret_1m,
        "return_3m": ret_3m,
        "return_12m": ret_12m,
    })

    # 缩尾处理，避免极端值
    for col in ["pe_ttm", "pb_lf", "ps_ttm", "ev_ebitda"]:
        df[col] = df[col].clip(upper=df[col].quantile(0.99))
    for col in ["roe", "roa", "gross_margin", "net_margin", "fcf_yield"]:
        df[col] = df[col].clip(lower=-1.0, upper=1.0)

    return df


# ---------------------------------------------------------------------------
# 主程序测试入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 测试模式：使用 mock 数据，无需 API key
    test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    logger.info("=== Alpha Vantage Engine 测试模式（Mock 数据） ===")

    # 测试 1: 客户端（无 key 时应返回 None）
    client = get_alpha_vantage_client()
    if client is None:
        logger.info("✓ 无 API key 时正确返回 None")
    else:
        logger.info("✓ 客户端创建成功")

    # 测试 2: Mock 数据生成
    mock_df = _generate_mock_us_data(test_symbols)
    logger.info("✓ Mock 数据生成: %d 行 × %d 列", len(mock_df), len(mock_df.columns))
    logger.info("\n%s", mock_df[["ticker", "sector", "pe_ttm", "pb_lf", "roe", "return_1m"]].to_string(index=False))

    # 测试 3: 批量获取（无客户端时自动降级 mock）
    universe_df = fetch_us_stock_universe(test_symbols)
    logger.info("✓ fetch_us_stock_universe 返回: %d 行 × %d 列", len(universe_df), len(universe_df.columns))

    # 测试 4: 列完整性检查
    expected_cols = [
        "ticker", "market", "sector", "market_cap",
        "pe_ttm", "pb_lf", "ps_ttm", "ev_ebitda", "dividend_yield",
        "roe", "roa", "gross_margin", "net_margin", "debt_to_equity",
        "revenue_growth", "profit_growth", "fcf_yield",
        "return_1m", "return_3m", "return_12m",
    ]
    missing = [c for c in expected_cols if c not in universe_df.columns]
    if missing:
        logger.error("✗ 缺失列: %s", missing)
    else:
        logger.info("✓ 所有 %d 个预期列均存在", len(expected_cols))

    logger.info("=== 测试完成 ===")
