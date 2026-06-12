#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini-GRP 统一数据获取与本地缓存模块

为 Mini-GRP 量化选股系统提供统一的数据接口和本地缓存层，支持：
- A 股：Tushare Pro / akshare / mock 三级降级
- 美股：Alpha Vantage / yfinance / mock 三级降级
- 跨市场：CN / US / HK / JP / KR 统一数据对齐
- 本地 SQLite 缓存：按数据源分子目录，WAL 模式保证并发安全

作者: Quant Dev
日期: 2025-06-11
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logger = logging.getLogger("unified_data_fetcher")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class DataSourceUnavailable(RuntimeError):
    """Raised when research mode cannot obtain a genuine market data set."""

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# 统一输出列名（16 个基础因子 + 3 个预期差距因子）
UNIFIED_COLUMNS: List[str] = [
    "code",
    "name",
    "sw_industry_name",
    "pe_ttm",
    "pb_lf",
    "ps_ttm",
    "ev_ebitda",
    "dividend_yield",
    "roe_deducted",
    "roa",
    "gross_margin",
    "net_margin",
    "debt_to_equity",
    "revenue_yoy",
    "profit_yoy",
    "fcf_yield",
    "return_1m",
    "return_3m",
    "return_12m",
    "sue",
    "eps_revision",
    "rating_revision",
]

# A 股数据源字段映射（原始 -> 统一）
A_SHARE_FIELD_MAP: Dict[str, str] = {
    # 通用行情字段
    "代码": "code",
    "名称": "name",
    "行业": "sw_industry_name",
    # 估值
    "市盈率-动态": "pe_ttm",
    "市盈率": "pe_ttm",
    "市净率": "pb_lf",
    "市销率": "ps_ttm",
    # 财务质量
    "净资产收益率-摊薄(%)": "roe_deducted",
    "净资产收益率": "roe_deducted",
    "总资产报酬率(%)": "roa",
    "总资产报酬率": "roa",
    "销售毛利率(%)": "gross_margin",
    "销售毛利率": "gross_margin",
    "销售净利率(%)": "net_margin",
    "销售净利率": "net_margin",
    "资产负债率(%)": "debt_to_equity",
    "资产负债率": "debt_to_equity",
    # 成长
    "营收同比增长率": "revenue_yoy",
    "净利润同比增长率": "profit_yoy",
    # 其他
    "股息率": "dividend_yield",
    "自由现金流收益率": "fcf_yield",
    # Tushare 字段
    "ts_code": "code",
    "industry": "sw_industry_name",
    "pe": "pe_ttm",
    "pb": "pb_lf",
    "ps": "ps_ttm",
    "ev_ebitda": "ev_ebitda",
    "roe": "roe_deducted",
    "roa": "roa",
    "grossprofit_margin": "gross_margin",
    "netprofit_margin": "net_margin",
    "debt_to_assets": "debt_to_equity",
    "dv_ratio": "dividend_yield",
    "rev_yoy": "revenue_yoy",
    "profit_yoy": "profit_yoy",
    "q_sales_yoy": "revenue_yoy",
    "q_profit_yoy": "profit_yoy",
}

# 美股数据源字段映射（原始 -> 统一）
US_FIELD_MAP: Dict[str, str] = {
    "symbol": "code",
    "ticker": "code",
    "name": "name",
    "sector": "sw_industry_name",
    "industry": "sw_industry_name",
    "trailingPE": "pe_ttm",
    "priceToBook": "pb_lf",
    "priceToSalesTrailing12Months": "ps_ttm",
    "enterpriseToEbitda": "ev_ebitda",
    "returnOnEquity": "roe_deducted",
    "returnOnAssets": "roa",
    "grossMargins": "gross_margin",
    "profitMargins": "net_margin",
    "debtToEquity": "debt_to_equity",
    "revenueGrowth": "revenue_yoy",
    "earningsGrowth": "profit_yoy",
    "dividendYield": "dividend_yield",
    "freeCashflowYield": "fcf_yield",
}

# 预期差距字段映射
EXPECTATION_FIELD_MAP: Dict[str, str] = {
    "sue": "sue",
    "eps_revision": "eps_revision",
    "rating_revision": "rating_revision",
    "std_eps_rev": "eps_revision",
    "rating_change": "rating_revision",
    "surprise": "sue",
}

# 市场配置
MARKET_CONFIG: Dict[str, Dict[str, Any]] = {
    "CN": {"currency": "CNY", "data_source": "tushare"},
    "US": {"currency": "USD", "data_source": "alpha_vantage"},
    "HK": {"currency": "HKD", "data_source": "yfinance"},
    "JP": {"currency": "JPY", "data_source": "yfinance"},
    "KR": {"currency": "KRW", "data_source": "yfinance"},
}


# ---------------------------------------------------------------------------
# 本地 SQLite 缓存层
# ---------------------------------------------------------------------------

class DataCache:
    """本地 SQLite 缓存，用于存储 API 响应避免重复请求。

    缓存策略：
    - 财务数据：缓存 7 天（变化不频繁）
    - 行情数据：缓存 1 天（每日更新）
    - 预期数据：缓存 1 天（每日更新）

    并发安全：使用 SQLite WAL 模式 + 线程级写入锁。
    """

    def __init__(self, cache_dir: str = "./cache") -> None:
        """初始化缓存。

        Args:
            cache_dir: 缓存根目录，默认当前目录下的 cache 文件夹。
                       子目录按数据源自动创建，如 ``./cache/tushare/``。
        """
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 按数据源创建子目录
        for source in ("tushare", "akshare", "alpha_vantage", "yfinance", "mock"):
            (self.cache_dir / source).mkdir(exist_ok=True)

        self._db_path = self.cache_dir / "cache_meta.db"
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """初始化 SQLite 元数据库，启用 WAL 模式。"""
        with sqlite3.connect(self._db_path, timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    data_source TEXT NOT NULL,
                    market TEXT NOT NULL,
                    stock_code TEXT,
                    cache_date TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    max_age_hours INTEGER NOT NULL,
                    row_count INTEGER,
                    columns TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_source_market ON cache_entries(data_source, market)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_created ON cache_entries(created_at)"
            )
            conn.commit()
        logger.info("缓存数据库初始化完成: %s", self._db_path)

    def _get_db_path(self, data_source: str) -> Path:
        """获取指定数据源的 SQLite 数据文件路径。"""
        return self.cache_dir / data_source / "data.db"

    def _build_key(self, data_source: str, market: str, stock_code: Optional[str], date_str: Optional[str] = None) -> str:
        """构建缓存键。

        格式: ``{data_source}_{market}_{stock_code}_{date}``
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")
        code_part = stock_code or "ALL"
        return f"{data_source}_{market}_{code_part}_{date_str}"

    def get(self, key: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        """从缓存获取数据，如果过期或不存在返回 None。

        Args:
            key: 缓存键。
            max_age_hours: 最大缓存年龄（小时），默认 24。

        Returns:
            缓存的 DataFrame，或 None（不存在/已过期）。
        """
        try:
            with sqlite3.connect(self._db_path, timeout=30.0) as meta_conn:
                meta_conn.row_factory = sqlite3.Row
                cur = meta_conn.execute(
                    "SELECT * FROM cache_entries WHERE key = ?", (key,)
                )
                row = cur.fetchone()

            if row is None:
                return None

            created_at = row["created_at"]
            data_source = row["data_source"]
            age_hours = (time.time() - created_at) / 3600.0

            if age_hours > max_age_hours:
                logger.debug("缓存键 %s 已过期 (%.1f 小时 > %d 小时)", key, age_hours, max_age_hours)
                return None

            # 从数据文件读取
            db_path = self._get_db_path(data_source)
            if not db_path.exists():
                return None

            with sqlite3.connect(db_path, timeout=30.0) as conn:
                df = pd.read_sql_query(f'SELECT * FROM "{key}"', conn)

            if df.empty:
                return None

            logger.debug("缓存命中: %s (%.1f 小时)", key, age_hours)
            return df

        except Exception as e:
            logger.warning("读取缓存失败 %s: %s", key, e)
            return None

    def set(self, key: str, data: pd.DataFrame, data_source: str = "mock", market: str = "CN", stock_code: Optional[str] = None, max_age_hours: int = 24) -> None:
        """将数据写入缓存。

        Args:
            key: 缓存键。
            data: 要缓存的 DataFrame。
            data_source: 数据源标识，用于确定子目录。
            market: 市场代码。
            stock_code: 股票代码（用于元数据记录）。
            max_age_hours: 缓存有效期（小时）。
        """
        if data is None or data.empty:
            logger.debug("数据为空，跳过缓存写入: %s", key)
            return

        with self._lock:
            try:
                db_path = self._get_db_path(data_source)
                db_path.parent.mkdir(parents=True, exist_ok=True)

                # 写入数据文件
                with sqlite3.connect(db_path, timeout=30.0) as conn:
                    data.to_sql(key, conn, if_exists="replace", index=False)

                # 更新元数据
                date_str = datetime.now().strftime("%Y%m%d")
                with sqlite3.connect(self._db_path, timeout=30.0) as meta_conn:
                    meta_conn.execute(
                        """
                        INSERT OR REPLACE INTO cache_entries
                        (key, data_source, market, stock_code, cache_date, created_at, max_age_hours, row_count, columns)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            key,
                            data_source,
                            market,
                            stock_code or "ALL",
                            date_str,
                            time.time(),
                            max_age_hours,
                            len(data),
                            json.dumps(list(data.columns)),
                        ),
                    )
                    meta_conn.commit()

                logger.debug("缓存写入完成: %s (%d 行)", key, len(data))

            except Exception as e:
                logger.warning("写入缓存失败 %s: %s", key, e)

    def clear_expired(self) -> None:
        """清理过期缓存（元数据及对应数据表）。"""
        with self._lock:
            try:
                now = time.time()
                with sqlite3.connect(self._db_path, timeout=30.0) as meta_conn:
                    meta_conn.row_factory = sqlite3.Row
                    cur = meta_conn.execute(
                        "SELECT key, data_source, created_at, max_age_hours FROM cache_entries"
                    )
                    expired = [
                        (row["key"], row["data_source"])
                        for row in cur.fetchall()
                        if (now - row["created_at"]) / 3600.0 > row["max_age_hours"]
                    ]

                if not expired:
                    logger.info("没有过期缓存需要清理")
                    return

                for key, data_source in expired:
                    try:
                        db_path = self._get_db_path(data_source)
                        if db_path.exists():
                            with sqlite3.connect(db_path, timeout=30.0) as conn:
                                conn.execute(f'DROP TABLE IF EXISTS "{key}"')
                                conn.commit()
                        with sqlite3.connect(self._db_path, timeout=30.0) as meta_conn:
                            meta_conn.execute(
                                "DELETE FROM cache_entries WHERE key = ?", (key,)
                            )
                            meta_conn.commit()
                        logger.debug("清理过期缓存: %s", key)
                    except Exception as e:
                        logger.warning("清理缓存失败 %s: %s", key, e)

                logger.info("清理完成，共删除 %d 条过期缓存", len(expired))

            except Exception as e:
                logger.error("清理过期缓存时出错: %s", e)

    def get_cache_stats(self) -> pd.DataFrame:
        """获取缓存统计信息。

        Returns:
            DataFrame，包含各数据源的缓存条目数、总行数、最新/最旧时间。
        """
        try:
            with sqlite3.connect(self._db_path, timeout=30.0) as meta_conn:
                df = pd.read_sql_query(
                    """
                    SELECT
                        data_source,
                        COUNT(*) as entries,
                        SUM(row_count) as total_rows,
                        MIN(created_at) as oldest,
                        MAX(created_at) as newest
                    FROM cache_entries
                    GROUP BY data_source
                    """,
                    meta_conn,
                )
            df["oldest"] = pd.to_datetime(df["oldest"], unit="s")
            df["newest"] = pd.to_datetime(df["newest"], unit="s")
            return df
        except Exception as e:
            logger.warning("获取缓存统计失败: %s", e)
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# Mock 数据生成器（与现有代码风格一致）
# ---------------------------------------------------------------------------

def _generate_mock_a_share(n: int = 100) -> pd.DataFrame:
    """生成 A 股模拟数据。"""
    logger.info("生成 %d 只 A 股模拟数据...", n)
    rng = np.random.default_rng(seed=42)
    industries = [
        "银行", "非银金融", "医药生物", "电子", "食品饮料",
        "电力设备", "计算机", "汽车", "化工", "机械设备",
        "家用电器", "通信", "房地产", "有色金属", "传媒",
        "交通运输", "农林牧渔", "建筑装饰", "钢铁", "采掘",
    ]

    codes = []
    for i in range(n):
        prefix = "600" if i < n // 2 else "300" if i < 3 * n // 4 else "000"
        codes.append(f"{prefix}{rng.integers(100, 999):03d}")

    df = pd.DataFrame({
        "code": codes,
        "name": [f"股票_{i+1}" for i in range(n)],
        "sw_industry_name": rng.choice(industries, size=n),
        "pe_ttm": rng.lognormal(3, 0.5, n),
        "pb_lf": rng.lognormal(0.5, 0.5, n),
        "ps_ttm": rng.lognormal(1, 0.6, n),
        "ev_ebitda": rng.lognormal(2.5, 0.6, n),
        "dividend_yield": rng.exponential(2, n),
        "roe_deducted": rng.normal(10, 8, n),
        "roa": rng.normal(5, 4, n),
        "gross_margin": rng.normal(30, 15, n),
        "net_margin": rng.normal(12, 10, n),
        "debt_to_equity": rng.lognormal(4, 0.5, n),
        "revenue_yoy": rng.normal(15, 20, n),
        "profit_yoy": rng.normal(12, 25, n),
        "fcf_yield": rng.normal(3, 2, n),
        "return_1m": rng.normal(0, 8, n),
        "return_3m": rng.normal(0, 15, n),
        "return_12m": rng.normal(10, 35, n),
        "sue": rng.normal(0, 1, n),
        "eps_revision": rng.normal(0, 0.05, n),
        "rating_revision": rng.normal(0, 0.1, n),
    })
    return df


def _generate_mock_us_data(n: int = 100) -> pd.DataFrame:
    """生成美股模拟数据。"""
    logger.info("生成 %d 只美股模拟数据...", n)
    rng = np.random.default_rng(seed=43)
    sectors = [
        "Technology", "Healthcare", "Financials", "Consumer Discretionary",
        "Communication Services", "Industrials", "Energy", "Materials",
        "Utilities", "Real Estate", "Consumer Staples",
    ]

    names = [
        "ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA", "ETA", "THETA",
        "IOTA", "KAPPA", "LAMBDA", "MU", "NU", "XI", "OMICRON", "PI", "RHO",
        "SIGMA", "TAU", "UPSILON", "PHI", "CHI", "PSI", "OMEGA",
    ]
    tickers = [f"{rng.choice(names)}{rng.integers(1, 99):02d}" for _ in range(n)]

    df = pd.DataFrame({
        "code": tickers,
        "name": [f"Mock_US_{i+1}" for i in range(n)],
        "sw_industry_name": rng.choice(sectors, size=n),
        "pe_ttm": rng.lognormal(3, 0.4, n),
        "pb_lf": rng.lognormal(0.6, 0.4, n),
        "ps_ttm": rng.lognormal(1.2, 0.5, n),
        "ev_ebitda": rng.lognormal(2.3, 0.5, n),
        "dividend_yield": rng.exponential(1.5, n),
        "roe_deducted": rng.normal(15, 10, n),
        "roa": rng.normal(7, 5, n),
        "gross_margin": rng.normal(40, 15, n),
        "net_margin": rng.normal(15, 12, n),
        "debt_to_equity": rng.lognormal(3.8, 0.6, n),
        "revenue_yoy": rng.normal(12, 18, n),
        "profit_yoy": rng.normal(10, 22, n),
        "fcf_yield": rng.normal(4, 2.5, n),
        "return_1m": rng.normal(0, 8, n),
        "return_3m": rng.normal(0, 15, n),
        "return_12m": rng.normal(10, 35, n),
        "sue": rng.normal(0, 1, n),
        "eps_revision": rng.normal(0, 0.05, n),
        "rating_revision": rng.normal(0, 0.1, n),
    })
    return df


def _generate_mock_expectation(stock_codes: List[str]) -> pd.DataFrame:
    """生成模拟预期差距数据。"""
    rng = np.random.default_rng(seed=44)
    n = len(stock_codes)
    return pd.DataFrame({
        "code": stock_codes,
        "sue": rng.normal(0, 1, n),
        "eps_revision": rng.normal(0, 0.05, n),
        "rating_revision": rng.normal(0, 0.1, n),
    })


# ---------------------------------------------------------------------------
# 字段映射与对齐工具
# ---------------------------------------------------------------------------

def _map_columns(df: pd.DataFrame, field_map: Dict[str, str]) -> pd.DataFrame:
    """将 DataFrame 的列名按映射表转换为统一名称。

    只映射存在的列，不存在的列保留原样。
    """
    if df.empty:
        return df
    rename_map = {k: v for k, v in field_map.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _fill_missing_by_industry(df: pd.DataFrame, factor_cols: List[str], industry_col: str = "sw_industry_name") -> pd.DataFrame:
    """使用行业均值填充缺失值（与 factor_engine.py 一致）。

    对于每个因子列，按行业计算均值，然后用行业均值填充该行业的缺失值。
    如果整个行业的值都缺失，则使用全市场均值填充，最后补 0。
    """
    result = df.copy()

    if industry_col not in result.columns:
        logger.warning("行业列 '%s' 不存在，使用全局均值填充", industry_col)
        for col in factor_cols:
            if col in result.columns:
                result[col] = result[col].fillna(result[col].mean())
                result[col] = result[col].fillna(0)
        return result

    for col in factor_cols:
        if col not in result.columns:
            continue

        # 按行业均值填充
        industry_means = result.groupby(industry_col)[col].transform("mean")
        result[col] = result[col].fillna(industry_means)

        # 如果仍有缺失（整个行业都缺失），用全局均值填充
        global_mean = result[col].mean()
        result[col] = result[col].fillna(global_mean)

        # 如果还是缺失，填 0
        result[col] = result[col].fillna(0)

    return result


def _ensure_unified_columns(df: pd.DataFrame, market: str, currency: str) -> pd.DataFrame:
    """确保 DataFrame 包含所有统一列，缺失的列用 NaN 填充。

    同时添加 ``market`` 和 ``currency`` 列。
    """
    result = df.copy()
    result["market"] = market
    result["currency"] = currency

    for col in UNIFIED_COLUMNS:
        if col not in result.columns:
            result[col] = np.nan

    # 调整列顺序
    meta_cols = ["market", "currency"]
    ordered = meta_cols + [c for c in UNIFIED_COLUMNS if c in result.columns]
    # 保留其他列
    other_cols = [c for c in result.columns if c not in ordered]
    return result[ordered + other_cols]


def _annotate_source(
    df: pd.DataFrame,
    source: str,
    is_mock: bool,
    max_stocks: Optional[int] = None,
) -> pd.DataFrame:
    """Attach provenance fields before data leaves the adapter boundary."""
    result = df.copy()
    if max_stocks is not None:
        result = result.head(max_stocks).copy()
    result["data_source"] = source
    result["is_mock"] = bool(is_mock)
    result["as_of_date"] = datetime.now().date().isoformat()
    expectation_cols = ["sue", "eps_revision", "rating_revision"]
    result["expectation_source"] = (
        "synthetic_demo" if is_mock else "provider_or_unavailable"
    )
    result["factor_coverage"] = result[
        [c for c in UNIFIED_COLUMNS if c not in {"code", "name", "sw_industry_name"}]
    ].notna().mean(axis=1)
    if not is_mock and not result[expectation_cols].notna().any().any():
        result["expectation_source"] = "unavailable"
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 统一 A 股数据接口
# ---------------------------------------------------------------------------

def fetch_a_share_data(stock_codes: Optional[List[str]] = None,
                       use_tushare: bool = True,
                       use_cache: bool = True,
                       cache: Optional[DataCache] = None,
                       max_stocks: Optional[int] = None,
                       allow_mock: bool = True) -> pd.DataFrame:
    """统一获取 A 股完整数据（财务 + 价格 + 行业 + 预期差距）。

    数据获取优先级：
    1. Tushare Pro（财务数据 + 分析师预期）
    2. akshare（fallback）
    3. mock 数据（最终 fallback）

    Args:
        stock_codes: 股票代码列表，如 ``['000001', '600000']``。
                     为 None 时获取全市场数据（受 API 限制）。
        use_tushare: 是否优先尝试 Tushare Pro。
        use_cache: 是否使用本地缓存。
        cache: 外部传入的 ``DataCache`` 实例；为 None 时自动创建。

    Returns:
        DataFrame with columns:
            - code, name, sw_industry_name
            - pe_ttm, pb_lf, ps_ttm, ev_ebitda, dividend_yield
            - roe_deducted, roa, gross_margin, net_margin, debt_to_equity
            - revenue_yoy, profit_yoy, fcf_yield
            - return_1m, return_3m, return_12m
            - sue, eps_revision, rating_revision
    """
    if cache is None and use_cache:
        cache = DataCache()

    date_str = datetime.now().strftime("%Y%m%d")
    cache_key = cache._build_key("tushare" if use_tushare else "akshare", "CN", "ALL", date_str) if cache else ""

    # 1. 尝试缓存
    if use_cache and cache is not None:
        cached = cache.get(cache_key, max_age_hours=24)
        if cached is not None and not cached.empty:
            cached_is_mock = bool(cached.get("is_mock", pd.Series([True])).all())
            if allow_mock or not cached_is_mock:
                logger.info("A 股数据缓存命中: %s", cache_key)
                return _annotate_source(
                    cached,
                    str(cached.get("data_source", pd.Series(["cache"])).iloc[0]),
                    cached_is_mock,
                    max_stocks,
                )

    # 2. 尝试 Tushare Pro
    if use_tushare:
        try:
            import tushare as ts
            token = os.environ.get("TUSHARE_TOKEN", "")
            api_url = os.environ.get("TUSHARE_API_URL", "")
            if token:
                pro = ts.pro_api(token)
                if api_url:
                    pro._DataApi__http_url = api_url
                    logger.info("使用自定义 Tushare API 地址: %s", api_url)
                logger.info("尝试通过 Tushare Pro 获取 A 股数据...")

                # 获取股票基础信息
                df_basic = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,industry")
                if df_basic is not None and not df_basic.empty:
                    df_basic = _map_columns(df_basic, A_SHARE_FIELD_MAP)

                # 获取每日指标（估值 + 部分财务）
                df_daily = pro.daily_basic(trade_date=date_str)
                if df_daily is None or df_daily.empty:
                    # 尝试前一个交易日
                    trade_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
                    df_daily = pro.daily_basic(trade_date=trade_date)

                if df_daily is not None and not df_daily.empty:
                    df_daily = _map_columns(df_daily, A_SHARE_FIELD_MAP)

                # 获取财务数据
                df_fin = pro.fina_indicator(period=str(int(date_str[:4]) - 1))
                if df_fin is not None and not df_fin.empty:
                    df_fin = _map_columns(df_fin, A_SHARE_FIELD_MAP)

                # 合并
                result = df_basic.copy() if df_basic is not None else pd.DataFrame()
                for merge_df in (df_daily, df_fin):
                    if merge_df is not None and not merge_df.empty and not result.empty:
                        result = result.merge(merge_df, on="code", how="left", suffixes=("", "_dup"))
                        # 去重列
                        dup_cols = [c for c in result.columns if c.endswith("_dup")]
                        result = result.drop(columns=dup_cols, errors="ignore")

                # 获取预期数据（分析师一致预期）
                try:
                    df_exp = pro.forecast(period=date_str[:4])
                    if df_exp is not None and not df_exp.empty:
                        df_exp = _map_columns(df_exp, EXPECTATION_FIELD_MAP)
                        if "code" in df_exp.columns:
                            result = result.merge(df_exp[["code", "sue", "eps_revision", "rating_revision"]], on="code", how="left")
                except Exception as e:
                    logger.warning("Tushare 预期数据获取失败: %s", e)

                if not result.empty:
                    result = _ensure_unified_columns(result, "CN", "CNY")
                    result = _annotate_source(result, "tushare", False, max_stocks)
                    if use_cache and cache is not None:
                        cache.set(cache_key, result, data_source="tushare", market="CN", max_age_hours=24)
                    logger.info("Tushare Pro 获取 A 股数据成功: %d 行", len(result))
                    return result
        except ImportError:
            logger.warning("Tushare Pro 未安装，跳过")
        except Exception as e:
            logger.warning("Tushare Pro 获取失败: %s，降级到 akshare", e)

    # 3. 尝试 akshare
    try:
        import akshare as ak
        logger.info("尝试通过 akshare 获取 A 股数据...")

        # 获取全市场快照
        spot_df = ak.stock_zh_a_spot_em()
        if spot_df is not None and not spot_df.empty:
            spot_df = _map_columns(spot_df, A_SHARE_FIELD_MAP)

            # 过滤指定股票
            if stock_codes:
                spot_df = spot_df[spot_df["code"].isin(stock_codes)]

            # 获取详细财务（限制数量避免超时）
            if stock_codes and len(stock_codes) <= 50:
                fin_records = []
                for code in stock_codes:
                    try:
                        detail = ak.stock_financial_analysis_indicator(symbol=code, start_year=str(datetime.now().year - 2))
                        if detail is not None and not detail.empty:
                            latest = detail.iloc[0]
                            fin_records.append({
                                "code": code,
                                "roe_deducted": pd.to_numeric(latest.get("净资产收益率-摊薄(%)"), errors="coerce"),
                                "roa": pd.to_numeric(latest.get("总资产报酬率(%)"), errors="coerce"),
                                "gross_margin": pd.to_numeric(latest.get("销售毛利率(%)"), errors="coerce"),
                                "net_margin": pd.to_numeric(latest.get("销售净利率(%)"), errors="coerce"),
                                "debt_to_equity": pd.to_numeric(latest.get("资产负债率(%)"), errors="coerce"),
                            })
                        time.sleep(0.2)
                    except Exception:
                        continue
                if fin_records:
                    fin_df = pd.DataFrame(fin_records)
                    spot_df = spot_df.merge(fin_df, on="code", how="left")

            # 获取价格数据（动量）
            if stock_codes and len(stock_codes) <= 100:
                price_records = []
                end_date = datetime.now().strftime("%Y%m%d")
                start_date = (datetime.now() - timedelta(days=400)).strftime("%Y%m%d")
                for code in stock_codes[:50]:  # 限制数量
                    try:
                        hist = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
                        if hist is not None and not hist.empty and len(hist) >= 30:
                            hist["日期"] = pd.to_datetime(hist["日期"])
                            hist = hist.sort_values("日期")
                            closes = hist["收盘"].astype(float)
                            latest = closes.iloc[-1]
                            info = {"code": code}
                            for period_name, period_days in (("1m", 22), ("3m", 66), ("12m", 250)):
                                if len(closes) >= period_days:
                                    past = closes.iloc[-period_days]
                                    info[f"return_{period_name}"] = (latest / past - 1) * 100
                                else:
                                    info[f"return_{period_name}"] = np.nan
                            price_records.append(info)
                        time.sleep(0.1)
                    except Exception:
                        continue
                if price_records:
                    price_df = pd.DataFrame(price_records)
                    spot_df = spot_df.merge(price_df, on="code", how="left")

            # 获取行业分类
            try:
                industry_df = ak.stock_board_industry_name_ths()
                if industry_df is not None and not industry_df.empty:
                    all_stocks = []
                    for _, row in industry_df.head(30).iterrows():
                        try:
                            industry_name = row.get("名称", "")
                            stocks = ak.stock_board_industry_cons_ths(symbol=industry_name)
                            if stocks is not None and not stocks.empty and "代码" in stocks.columns:
                                for _, s in stocks.iterrows():
                                    all_stocks.append({"code": s["代码"], "sw_industry_name": industry_name})
                            time.sleep(0.15)
                        except Exception:
                            continue
                    if all_stocks:
                        ind_df = pd.DataFrame(all_stocks).drop_duplicates("code")
                        if "sw_industry_name" in spot_df.columns:
                            spot_df = spot_df.drop(columns=["sw_industry_name"], errors="ignore")
                        spot_df = spot_df.merge(ind_df, on="code", how="left")
            except Exception as e:
                logger.warning("获取行业分类失败: %s", e)

            if not spot_df.empty:
                spot_df = _ensure_unified_columns(spot_df, "CN", "CNY")
                spot_df = _annotate_source(spot_df, "akshare", False, max_stocks)
                if use_cache and cache is not None:
                    cache.set(cache_key, spot_df, data_source="akshare", market="CN", max_age_hours=24)
                logger.info("akshare 获取 A 股数据成功: %d 行", len(spot_df))
                return spot_df
    except ImportError:
        logger.warning("akshare 未安装，跳过")
    except Exception as e:
        logger.warning("akshare 获取失败: %s，降级到 mock", e)

    # 4. Mock 数据 fallback
    if not allow_mock:
        raise DataSourceUnavailable(
            "A-share research data is unavailable. Configure TUSHARE_TOKEN or install akshare; synthetic fallback is disabled."
        )
    logger.info("使用 mock 数据作为 A 股最终 fallback")
    mock_df = _generate_mock_a_share(n=len(stock_codes) if stock_codes else 100)
    if stock_codes:
        mock_df = mock_df[mock_df["code"].isin(stock_codes)]
    mock_df = _ensure_unified_columns(mock_df, "CN", "CNY")
    mock_df = _fill_missing_by_industry(mock_df, [c for c in UNIFIED_COLUMNS if c not in ("code", "name", "sw_industry_name")])
    mock_df = _annotate_source(mock_df, "synthetic_demo", True, max_stocks)
    if use_cache and cache is not None:
        cache.set(cache_key, mock_df, data_source="mock", market="CN", max_age_hours=24)
    return mock_df


# ---------------------------------------------------------------------------
# 统一美股数据接口
# ---------------------------------------------------------------------------

def fetch_us_data(symbols: Optional[List[str]] = None,
                  use_alpha_vantage: bool = True,
                  use_cache: bool = True,
                  cache: Optional[DataCache] = None,
                  max_stocks: Optional[int] = None,
                  allow_mock: bool = True) -> pd.DataFrame:
    """统一获取美股完整数据。

    数据获取优先级：
    1. Alpha Vantage（结构化财务数据）
    2. yfinance（fallback）
    3. mock 数据（最终 fallback）

    返回的 DataFrame 列名与 A 股对齐（统一字段名）。

    Args:
        symbols: 美股代码列表，如 ``['AAPL', 'MSFT']``。
                 为 None 时使用预设的 S&P 500 代表性股票。
        use_alpha_vantage: 是否优先尝试 Alpha Vantage。
        use_cache: 是否使用本地缓存。
        cache: 外部传入的 ``DataCache`` 实例；为 None 时自动创建。

    Returns:
        DataFrame，列名与 ``fetch_a_share_data`` 统一。
    """
    if cache is None and use_cache:
        cache = DataCache()

    date_str = datetime.now().strftime("%Y%m%d")
    cache_key = cache._build_key("alpha_vantage" if use_alpha_vantage else "yfinance", "US", "ALL", date_str) if cache else ""

    # 1. 尝试缓存
    if use_cache and cache is not None:
        cached = cache.get(cache_key, max_age_hours=24)
        if cached is not None and not cached.empty:
            cached_is_mock = bool(cached.get("is_mock", pd.Series([True])).all())
            if allow_mock or not cached_is_mock:
                logger.info("美股数据缓存命中: %s", cache_key)
                return _annotate_source(
                    cached,
                    str(cached.get("data_source", pd.Series(["cache"])).iloc[0]),
                    cached_is_mock,
                    max_stocks,
                )

    # 默认股票池
    if symbols is None:
        symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
            "AVGO", "WMT", "JPM", "LLY", "V", "UNH", "XOM", "MA", "PG", "JNJ",
            "HD", "MRK", "CVX", "PEP", "KO", "ABBV", "BAC", "COST", "TMO",
            "DIS", "ABT", "ADBE", "CRM", "ACN", "VZ", "WFC", "DHR", "TXN",
            "PM", "NKE", "RTX", "INTC", "UPS", "NEE", "QCOM", "MS", "LIN",
            "AMD", "HON", "INTU", "SPGI", "GS", "CAT",
        ]

    # 2. 尝试 Alpha Vantage
    if use_alpha_vantage:
        try:
            from alpha_vantage.fundamentaldata import FundamentalData
            api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
            if api_key:
                logger.info("尝试通过 Alpha Vantage 获取美股数据...")
                fd = FundamentalData(key=api_key, output_format="pandas")
                records = []
                for sym in symbols[:25]:  # 免费版限制
                    try:
                        # 公司概况
                        overview, _ = fd.get_company_overview(sym)
                        if overview is not None and not overview.empty:
                            row = overview.iloc[0].to_dict()
                            row["symbol"] = sym
                            records.append(row)
                        time.sleep(0.5)  # 避免限流
                    except Exception:
                        continue
                if records:
                    df = pd.DataFrame(records)
                    df = _map_columns(df, US_FIELD_MAP)
                    df = _map_columns(df, A_SHARE_FIELD_MAP)  # 二次映射兜底
                    df = df.rename(columns={"symbol": "code"})

                    # 获取价格数据（动量）
                    try:
                        from alpha_vantage.timeseries import TimeSeries
                        ts = TimeSeries(key=api_key, output_format="pandas")
                        price_records = []
                        for sym in symbols[:25]:
                            try:
                                data, _ = ts.get_daily_adjusted(symbol=sym, outputsize="full")
                                if data is not None and not data.empty:
                                    data = data.sort_index()
                                    close = data["5. adjusted close"].astype(float)
                                    latest = close.iloc[-1]
                                    price_records.append({
                                        "code": sym,
                                        "return_1m": (latest / close.iloc[-22] - 1) * 100 if len(close) >= 22 else np.nan,
                                        "return_3m": (latest / close.iloc[-66] - 1) * 100 if len(close) >= 66 else np.nan,
                                        "return_12m": (latest / close.iloc[-252] - 1) * 100 if len(close) >= 252 else np.nan,
                                    })
                                time.sleep(0.5)
                            except Exception:
                                continue
                        if price_records:
                            price_df = pd.DataFrame(price_records)
                            df = df.merge(price_df, on="code", how="left")
                    except Exception as e:
                        logger.warning("Alpha Vantage 价格数据获取失败: %s", e)

                    if not df.empty:
                        df = _ensure_unified_columns(df, "US", "USD")
                        df = _annotate_source(df, "alpha_vantage", False, max_stocks)
                        if use_cache and cache is not None:
                            cache.set(cache_key, df, data_source="alpha_vantage", market="US", max_age_hours=24)
                        logger.info("Alpha Vantage 获取美股数据成功: %d 行", len(df))
                        return df
        except ImportError:
            logger.warning("alpha_vantage 未安装，跳过")
        except Exception as e:
            logger.warning("Alpha Vantage 获取失败: %s，降级到 yfinance", e)

    # 3. 尝试 yfinance
    try:
        import yfinance as yf
        logger.info("尝试通过 yfinance 获取美股数据...")

        records = []
        for sym in symbols:
            try:
                ticker = yf.Ticker(sym)
                info = ticker.info or {}
                records.append({
                    "code": sym,
                    "name": info.get("longName", sym),
                    "sw_industry_name": info.get("sector", info.get("industry", "Unknown")),
                    "pe_ttm": info.get("trailingPE", np.nan),
                    "pb_lf": info.get("priceToBook", np.nan),
                    "ps_ttm": info.get("priceToSalesTrailing12Months", np.nan),
                    "ev_ebitda": info.get("enterpriseToEbitda", np.nan),
                    "dividend_yield": (info.get("dividendYield", 0) or 0) * 100,
                    "roe_deducted": info.get("returnOnEquity", np.nan),
                    "roa": info.get("returnOnAssets", np.nan),
                    "gross_margin": (info.get("grossMargins", 0) or 0) * 100,
                    "net_margin": (info.get("profitMargins", 0) or 0) * 100,
                    "debt_to_equity": info.get("debtToEquity", np.nan),
                    "revenue_yoy": (info.get("revenueGrowth", 0) or 0) * 100,
                    "profit_yoy": (info.get("earningsGrowth", 0) or 0) * 100,
                    "fcf_yield": np.nan,
                })
            except Exception:
                continue

        if records:
            df = pd.DataFrame(records)

            # 批量获取价格数据
            try:
                data = yf.download(tickers=symbols, period="1y", interval="1d", auto_adjust=True, progress=False, threads=True)
                if data is not None and not data.empty:
                    if len(symbols) == 1:
                        close = data["Close"].to_frame(symbols[0])
                    else:
                        close = data["Close"]
                    ret_1m = close.pct_change(21).iloc[-1]
                    ret_3m = close.pct_change(63).iloc[-1]
                    ret_12m = close.pct_change(252).iloc[-1]
                    price_df = pd.DataFrame({
                        "code": symbols,
                        "return_1m": ret_1m.reindex(symbols).fillna(0).values * 100,
                        "return_3m": ret_3m.reindex(symbols).fillna(0).values * 100,
                        "return_12m": ret_12m.reindex(symbols).fillna(0).values * 100,
                    })
                    df = df.merge(price_df, on="code", how="left")
            except Exception as e:
                logger.warning("yfinance 批量价格获取失败: %s", e)

            if not df.empty:
                df = _ensure_unified_columns(df, "US", "USD")
                df = _annotate_source(df, "yfinance", False, max_stocks)
                if use_cache and cache is not None:
                    cache.set(cache_key, df, data_source="yfinance", market="US", max_age_hours=24)
                logger.info("yfinance 获取美股数据成功: %d 行", len(df))
                return df
    except ImportError:
        logger.warning("yfinance 未安装，跳过")
    except Exception as e:
        logger.warning("yfinance 获取失败: %s，降级到 mock", e)

    # 4. Mock 数据 fallback
    if not allow_mock:
        raise DataSourceUnavailable(
            "US research data is unavailable. Configure ALPHA_VANTAGE_API_KEY or install yfinance; synthetic fallback is disabled."
        )
    logger.info("使用 mock 数据作为美股最终 fallback")
    mock_df = _generate_mock_us_data(n=len(symbols) if symbols else 100)
    if symbols:
        mock_df = mock_df[mock_df["code"].isin(symbols)]
    mock_df = _ensure_unified_columns(mock_df, "US", "USD")
    mock_df = _fill_missing_by_industry(mock_df, [c for c in UNIFIED_COLUMNS if c not in ("code", "name", "sw_industry_name")])
    mock_df = _annotate_source(mock_df, "synthetic_demo", True, max_stocks)
    if use_cache and cache is not None:
        cache.set(cache_key, mock_df, data_source="mock", market="US", max_age_hours=24)
    return mock_df


# ---------------------------------------------------------------------------
# 统一跨市场数据接口
# ---------------------------------------------------------------------------

def fetch_multi_market_data(markets: Optional[List[str]] = None,
                            max_stocks_per_market: int = 100,
                            use_cache: bool = True,
                            allow_mock: bool = True) -> pd.DataFrame:
    """获取多个市场的统一数据。

    Args:
        markets: 市场代码列表，如 ``['CN', 'US', 'HK']``。
        max_stocks_per_market: 每个市场最大股票数。
        use_cache: 是否使用本地缓存。

    Returns:
        DataFrame with columns:
            - ticker / code
            - name
            - market ('CN', 'US', etc.)
            - sector / sw_industry_name
            - currency
            - 所有 16+3 个因子（统一命名）
    """
    markets = markets or ["CN", "US"]
    cache = DataCache() if use_cache else None
    all_dfs: List[pd.DataFrame] = []

    for market in markets:
        market = market.upper()
        logger.info("=" * 50)
        logger.info("开始获取市场 %s 数据 (max=%d)...", market, max_stocks_per_market)

        try:
            if market == "CN":
                df = fetch_a_share_data(stock_codes=None, use_tushare=True, use_cache=use_cache, cache=cache, max_stocks=max_stocks_per_market, allow_mock=allow_mock)
            elif market == "US":
                df = fetch_us_data(symbols=None, use_alpha_vantage=True, use_cache=use_cache, cache=cache, max_stocks=max_stocks_per_market, allow_mock=allow_mock)
            elif market in ("HK", "JP", "KR"):
                # 复用 yfinance 适配器逻辑（通过 multi_market_engine 的适配器）
                try:
                    from multi_market_engine import YahooFinanceAdapter, generate_mock_stock_list, generate_mock_financials, generate_mock_price_data
                    adapter = YahooFinanceAdapter(market)
                    stock_list = adapter.get_stock_list(max_stocks=max_stocks_per_market)
                    financials = adapter.get_financials(stock_list)
                    price_data = adapter.get_price_data(stock_list)

                    # 合并并映射字段
                    base = financials.copy()
                    if "ticker" in price_data.columns:
                        price_cols = ["ticker", "RETURN_1M", "RETURN_3M", "RETURN_12M"]
                        price_cols = [c for c in price_cols if c in price_data.columns]
                        base = base.merge(price_data[price_cols], on="ticker", how="left")

                    # 统一列名
                    base = base.rename(columns={
                        "ticker": "code",
                        "sector": "sw_industry_name",
                        "RETURN_1M": "return_1m",
                        "RETURN_3M": "return_3m",
                        "RETURN_12M": "return_12m",
                    })

                    # 计算统一估值/质量/成长因子（从原始财务字段）
                    eps = 1e-8
                    if "market_cap" in base.columns:
                        base["pe_ttm"] = base["market_cap"] / (base.get("net_profit_ttm", 0) + eps)
                        base["pb_lf"] = base["market_cap"] / (base.get("book_value", 0) + eps)
                        base["ps_ttm"] = base["market_cap"] / (base.get("revenue_ttm", 0) + eps)
                        base["ev_ebitda"] = (base["market_cap"] + base.get("debt", 0).fillna(0) - base.get("cash", 0).fillna(0)) / (base.get("ebitda_ttm", 0) + eps)
                        base["dividend_yield"] = base.get("dividend_ttm", 0) / (base["market_cap"] + eps)
                        base["fcf_yield"] = base.get("fcf_ttm", 0) / (base["market_cap"] + eps)

                    if "net_profit_ttm" in base.columns and "equity" in base.columns:
                        base["roe_deducted"] = base["net_profit_ttm"] / (base["equity"] + eps)
                    if "net_profit_ttm" in base.columns and "total_assets" in base.columns:
                        base["roa"] = base["net_profit_ttm"] / (base["total_assets"] + eps)
                    if "gross_profit" in base.columns and "revenue_ttm" in base.columns:
                        base["gross_margin"] = base["gross_profit"] / (base["revenue_ttm"] + eps)
                    if "net_profit_ttm" in base.columns and "revenue_ttm" in base.columns:
                        base["net_margin"] = base["net_profit_ttm"] / (base["revenue_ttm"] + eps)
                    if "total_liabilities" in base.columns and "equity" in base.columns:
                        base["debt_to_equity"] = base["total_liabilities"] / (base["equity"] + eps)

                    base["revenue_yoy"] = base.get("revenue_yoy", np.nan)
                    base["profit_yoy"] = base.get("profit_yoy", np.nan)

                    # 预期差距（mock 补充）
                    exp_df = _generate_mock_expectation(base["code"].tolist())
                    base = base.merge(exp_df, on="code", how="left")

                    df = _ensure_unified_columns(base, market, MARKET_CONFIG.get(market, {}).get("currency", "USD"))
                    df = _fill_missing_by_industry(df, [c for c in UNIFIED_COLUMNS if c not in ("code", "name", "sw_industry_name")])
                except Exception as e:
                    if not allow_mock:
                        raise DataSourceUnavailable(
                            f"{market} research data is unavailable and synthetic fallback is disabled"
                        ) from e
                    logger.warning("市场 %s 适配器获取失败: %s，使用 mock", market, e)
                    df = _generate_mock_us_data(n=max_stocks_per_market)
                    df["code"] = df["code"] + f"_{market}"
                    df = _ensure_unified_columns(df, market, MARKET_CONFIG.get(market, {}).get("currency", "USD"))
                    df = _annotate_source(df, "synthetic_demo", True, max_stocks_per_market)
            else:
                logger.warning("不支持的市场: %s，跳过", market)
                continue

            if not df.empty:
                # 限制数量
                df = df.head(max_stocks_per_market)
                all_dfs.append(df)
                logger.info("市场 %s 获取成功: %d 行", market, len(df))

        except Exception as e:
            logger.error("获取市场 %s 数据时出错: %s", market, e)
            if not allow_mock:
                raise
            continue

    if not all_dfs:
        raise DataSourceUnavailable("No requested market produced a usable data set")

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info("=" * 50)
    logger.info("跨市场数据合并完成: %d 只股票, %d 列", len(combined), len(combined.columns))
    return combined


# ---------------------------------------------------------------------------
# 数据合并与对齐工具
# ---------------------------------------------------------------------------

def merge_with_expectation_data(financial_df: pd.DataFrame,
                                expectation_df: pd.DataFrame,
                                on: str = "code") -> pd.DataFrame:
    """将财务数据与预期差距数据合并。

    处理字段名映射、缺失值填充（行业均值填充）。

    Args:
        financial_df: 财务数据 DataFrame，必须包含 ``on`` 指定的列和 ``sw_industry_name``。
        expectation_df: 预期差距数据 DataFrame，必须包含 ``on`` 指定的列。
        on: 合并键，默认 ``code``。

    Returns:
        合并后的 DataFrame，缺失值已按行业均值填充。
    """
    if financial_df.empty:
        logger.warning("财务数据为空，直接返回预期数据")
        return expectation_df.copy()
    if expectation_df.empty:
        logger.warning("预期数据为空，直接返回财务数据")
        return financial_df.copy()

    # 字段映射
    expectation_df = _map_columns(expectation_df, EXPECTATION_FIELD_MAP)

    # 合并
    merged = financial_df.merge(expectation_df, on=on, how="left", suffixes=("", "_exp"))

    # 处理冲突列（保留原始财务数据，丢弃 _exp 重复列）
    dup_cols = [c for c in merged.columns if c.endswith("_exp")]
    merged = merged.drop(columns=dup_cols, errors="ignore")

    # 缺失值填充（行业均值）
    exp_cols = ["sue", "eps_revision", "rating_revision"]
    available_exp = [c for c in exp_cols if c in merged.columns]
    if available_exp:
        merged = _fill_missing_by_industry(merged, available_exp, industry_col="sw_industry_name")

    logger.info("财务与预期数据合并完成: %d 行, 新增预期列: %s", len(merged), available_exp)
    return merged


# ---------------------------------------------------------------------------
# 测试入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Mini-GRP 统一数据获取模块测试")
    print("=" * 70)

    # 1. 测试 DataCache
    print("\n[1/6] 测试 DataCache...")
    cache = DataCache(cache_dir="./cache_test")
    test_df = pd.DataFrame({
        "code": ["000001", "600000"],
        "name": ["平安银行", "浦发银行"],
        "pe_ttm": [8.5, 6.2],
    })
    cache.set("test_key_001", test_df, data_source="tushare", market="CN", max_age_hours=24)
    cached = cache.get("test_key_001", max_age_hours=24)
    if cached is not None:
        print(f"  缓存写入/读取成功: {len(cached)} 行")
    else:
        print("  缓存读取失败")
    print(f"  缓存统计:\n{cache.get_cache_stats().to_string()}")

    # 2. 测试 A 股数据获取
    print("\n[2/6] 测试 fetch_a_share_data()...")
    a_share_df = fetch_a_share_data(stock_codes=["000001", "600000"], use_tushare=True, use_cache=True, cache=cache)
    print(f"  获取到 {len(a_share_df)} 行 A 股数据")
    print(f"  列名: {a_share_df.columns.tolist()}")
    if not a_share_df.empty:
        print(f"  前 2 行:\n{a_share_df.head(2).to_string()}")

    # 3. 测试美股数据获取
    print("\n[3/6] 测试 fetch_us_data()...")
    us_df = fetch_us_data(symbols=["AAPL", "MSFT"], use_alpha_vantage=True, use_cache=True, cache=cache)
    print(f"  获取到 {len(us_df)} 行美股数据")
    print(f"  列名: {us_df.columns.tolist()}")
    if not us_df.empty:
        print(f"  前 2 行:\n{us_df.head(2).to_string()}")

    # 4. 测试跨市场数据获取
    print("\n[4/6] 测试 fetch_multi_market_data()...")
    multi_df = fetch_multi_market_data(markets=["CN", "US"], max_stocks_per_market=20, use_cache=True)
    print(f"  跨市场合并: {len(multi_df)} 行, {len(multi_df.columns)} 列")
    print(f"  市场分布: {multi_df['market'].value_counts().to_dict()}")
    print(f"  统一因子列: {[c for c in UNIFIED_COLUMNS if c in multi_df.columns]}")

    # 5. 测试数据合并与对齐
    print("\n[5/6] 测试 merge_with_expectation_data()...")
    mock_fin = pd.DataFrame({
        "code": ["000001", "600000", "000002"],
        "name": ["平安银行", "浦发银行", "万科A"],
        "sw_industry_name": ["银行", "银行", "房地产"],
        "pe_ttm": [8.5, 6.2, 5.1],
    })
    mock_exp = pd.DataFrame({
        "code": ["000001", "000002"],
        "sue": [0.5, -0.2],
        "eps_revision": [0.03, -0.01],
    })
    merged = merge_with_expectation_data(mock_fin, mock_exp, on="code")
    print(f"  合并后: {len(merged)} 行")
    print(f"  缺失值填充检查:\n{merged[['code', 'sue', 'eps_revision', 'rating_revision']].to_string()}")

    # 6. 测试缓存清理
    print("\n[6/6] 测试 clear_expired()...")
    cache.clear_expired()
    print(f"  清理后统计:\n{cache.get_cache_stats().to_string()}")

    print("\n" + "=" * 70)
    print("统一数据获取模块测试完成!")
    print("=" * 70)
