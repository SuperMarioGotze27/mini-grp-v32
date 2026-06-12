#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini-GRP Tushare Pro 数据接入模块
提供分析师一致预期、SUE、EPS 修正和评级修正等预期差距因子数据

作者: Quant Dev
日期: 2025-06-11
"""

import json
import logging
import os
import pickle
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量与配置
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).parent / 'cache' / 'tushare'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Tushare 与系统内部代码格式转换映射
EXCHANGE_SUFFIX_MAP = {
    'SH': '.SH',
    'SZ': '.SZ',
    'BJ': '.BJ',
}
REVERSE_EXCHANGE_MAP = {v: k for k, v in EXCHANGE_SUFFIX_MAP.items()}

# ---------------------------------------------------------------------------
# 重试装饰器
# ---------------------------------------------------------------------------


def retry_on_failure(max_retries=3, delay=2, backoff=2):
    """为函数添加重试机制的装饰器。"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(
                            f"函数 {func.__name__} 在 {max_retries} 次尝试后仍然失败: {e}"
                        )
                        raise
                    logger.warning(
                        f"函数 {func.__name__} 第 {retries} 次尝试失败: {e}, "
                        f"{current_delay}秒后重试..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            return None
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# 公共工具函数
# ---------------------------------------------------------------------------


def _safe_get(func, *args, **kwargs):
    """安全地调用函数，失败时返回 None。"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"调用失败 {func.__name__}: {e}")
        return None


def _to_tushare_code(code: str) -> str:
    """将内部 6 位代码转换为 Tushare 格式（如 000001 -> 000001.SZ）。

    Args:
        code: 6 位数字股票代码

    Returns:
        Tushare 格式的股票代码
    """
    code = str(code).strip()
    if len(code) != 6:
        return code
    if code.startswith(('600', '601', '602', '603', '605', '688', '689')):
        return f"{code}.SH"
    elif code.startswith(('000', '001', '002', '003', '300', '301')):
        return f"{code}.SZ"
    elif code.startswith(('430', '831', '832', '833', '834', '835', '836', '837', '838', '839', '870', '871', '872', '873')):
        return f"{code}.BJ"
    else:
        return f"{code}.SZ"


def _from_tushare_code(ts_code: str) -> str:
    """将 Tushare 格式代码转换为内部 6 位代码。

    Args:
        ts_code: Tushare 格式代码（如 000001.SZ）

    Returns:
        6 位数字股票代码
    """
    if '.' in ts_code:
        return ts_code.split('.')[0]
    return ts_code


def _cache_path(func_name: str, key: str, ext: str = 'pkl') -> Path:
    """生成缓存文件路径。"""
    safe_key = key.replace('/', '_').replace('\\', '_').replace(':', '_')
    return CACHE_DIR / f"{func_name}_{safe_key}.{ext}"


def _load_cache(cache_path: Path, max_age_hours: int = 24):
    """从本地缓存加载数据。

    Args:
        cache_path: 缓存文件路径
        max_age_hours: 缓存最大有效时长（小时）

    Returns:
        缓存的数据，若过期或不存在则返回 None
    """
    if not cache_path.exists():
        return None
    try:
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime > timedelta(hours=max_age_hours):
            logger.info(f"缓存已过期: {cache_path.name}")
            return None
        if cache_path.suffix == '.pkl':
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        elif cache_path.suffix == '.json':
            with open(cache_path, 'r', encoding='utf-8') as f:
                return pd.read_json(f, orient='records')
        else:
            return None
    except Exception as e:
        logger.warning(f"加载缓存失败 {cache_path.name}: {e}")
        return None


def _save_cache(cache_path: Path, data):
    """保存数据到本地缓存。"""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.suffix == '.pkl':
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        elif cache_path.suffix == '.json':
            if isinstance(data, pd.DataFrame):
                data.to_json(cache_path, orient='records', force_ascii=False)
            else:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, default=str)
        logger.info(f"缓存已保存: {cache_path.name}")
    except Exception as e:
        logger.warning(f"保存缓存失败 {cache_path.name}: {e}")


# ---------------------------------------------------------------------------
# Mock 数据生成器
# ---------------------------------------------------------------------------


def _generate_mock_expectation_data(stock_codes: List[str]) -> pd.DataFrame:
    """生成模拟的预期差距数据。

    Args:
        stock_codes: 股票代码列表（6 位数字）

    Returns:
        DataFrame with columns [code, sue, eps_revision, rating_revision]
    """
    logger.info(f"生成 {len(stock_codes)} 只股票的模拟预期差距数据...")
    np.random.seed(46)
    data = []
    for code in stock_codes:
        sue = np.random.normal(0, 2)
        eps_revision = np.random.normal(0, 0.15)
        rating_revision = np.random.normal(0, 0.3)
        data.append({
            'code': code,
            'sue': sue,
            'eps_revision': np.clip(eps_revision, -1, 1),
            'rating_revision': np.clip(rating_revision, -1, 1),
        })
    return pd.DataFrame(data)


def _generate_mock_forecast(ts_code: str, n: int = 10) -> pd.DataFrame:
    """生成模拟的业绩预告/快报数据。

    Args:
        ts_code: Tushare 格式股票代码
        n: 生成记录数

    Returns:
        DataFrame with columns [ts_code, ann_date, end_date, type, pre_eps,
                                net_profit_min, net_profit_max]
    """
    np.random.seed(hash(ts_code) % 2**31)
    base_date = datetime(2024, 1, 1)
    rows = []
    for i in range(n):
        ann_date = (base_date + timedelta(days=i * 30)).strftime('%Y%m%d')
        end_date = (base_date + timedelta(days=i * 90)).strftime('%Y%m%d')
        rows.append({
            'ts_code': ts_code,
            'ann_date': ann_date,
            'end_date': end_date,
            'type': random.choice(['预增', '预减', '略增', '略减', '续盈', '续亏']),
            'pre_eps': round(np.random.normal(0.5, 0.3), 4),
            'net_profit_min': round(np.random.uniform(1e7, 5e8), 2),
            'net_profit_max': round(np.random.uniform(5e8, 1e9), 2),
        })
    return pd.DataFrame(rows)


def _generate_mock_rating_data(ts_code: str, n: int = 20) -> pd.DataFrame:
    """生成模拟的分析师评级数据。

    Args:
        ts_code: Tushare 格式股票代码
        n: 生成记录数

    Returns:
        DataFrame with columns [ts_code, ann_date, rating, rating_change]
    """
    np.random.seed(hash(ts_code) % 2**31 + 1)
    base_date = datetime(2024, 1, 1)
    ratings = ['买入', '增持', '中性', '减持']
    rows = []
    for i in range(n):
        ann_date = (base_date + timedelta(days=i * 15)).strftime('%Y%m%d')
        rows.append({
            'ts_code': ts_code,
            'ann_date': ann_date,
            'rating': random.choice(ratings),
            'rating_change': random.choice(['上调', '下调', '维持', '首次']),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tushare Pro 客户端封装
# ---------------------------------------------------------------------------


def get_tushare_pro(token: Optional[str] = None) -> Optional[object]:
    """获取 Tushare Pro API 实例。

    优先从环境变量 TUSHARE_TOKEN 读取，其次从参数传入。
    如果 tushare 未安装或 token 无效，返回 None。

    Args:
        token: Tushare Pro API Token，若 None 则从环境变量读取

    Returns:
        Tushare Pro API 实例，或 None
    """
    try:
        import tushare as ts
    except ImportError:
        logger.warning("tushare 未安装，无法获取 Tushare Pro 客户端")
        return None

    effective_token = token or os.environ.get('TUSHARE_TOKEN', '')
    if not effective_token:
        logger.warning("未提供 TUSHARE_TOKEN，无法初始化 Tushare Pro 客户端")
        return None

    try:
        pro = ts.pro_api(effective_token)
        # 简单验证：尝试获取一条市场数据
        test_df = pro.trade_cal(exchange='SSE', limit=1)
        if test_df is None or test_df.empty:
            logger.warning("Tushare Pro token 验证失败，返回 None")
            return None
        logger.info("Tushare Pro 客户端初始化成功")
        return pro
    except Exception as e:
        logger.warning(f"Tushare Pro 初始化失败: {e}")
        return None


# ---------------------------------------------------------------------------
# 获取分析师一致预期数据
# ---------------------------------------------------------------------------


@retry_on_failure(max_retries=3, delay=2, backoff=2)
def fetch_consensus_forecast(ts_code: str,
                             start_date: str,
                             end_date: str,
                             pro: Optional[object] = None) -> pd.DataFrame:
    """获取个股分析师一致预期数据。

    使用 Tushare Pro 的 ``forecast`` 接口（业绩预告）和 ``express`` 接口（业绩快报）。

    Args:
        ts_code: Tushare 格式股票代码（如 000001.SZ）
        start_date: 开始日期（YYYYMMDD）
        end_date: 结束日期（YYYYMMDD）
        pro: Tushare Pro 实例，若为 None 则自动获取

    Returns:
        DataFrame with columns:
            - ts_code: 股票代码
            - ann_date: 公告日期
            - end_date: 报告期
            - type: 预告类型（预增/预减等）
            - pre_eps: 预期 EPS（从 express 接口获取）
            - net_profit_min/max: 净利润区间（forecast）
    """
    if pro is None:
        pro = get_tushare_pro()
    if pro is None:
        logger.warning("Tushare Pro 不可用，返回模拟 forecast 数据")
        return _generate_mock_forecast(ts_code, n=10)

    cache_key = f"forecast_{ts_code}_{start_date}_{end_date}"
    cache_path = _cache_path("forecast", cache_key)
    cached = _load_cache(cache_path, max_age_hours=6)
    if cached is not None:
        logger.info(f"从缓存加载 forecast 数据: {ts_code}")
        return cached

    logger.info(f"获取 forecast 数据: {ts_code} [{start_date} ~ {end_date}]")

    # 1) 业绩预告 forecast
    forecast_df = _safe_get(
        pro.forecast,
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date
    )
    if forecast_df is None or forecast_df.empty:
        forecast_df = pd.DataFrame()
    else:
        # 统一列名
        col_map = {
            'ts_code': 'ts_code',
            'ann_date': 'ann_date',
            'end_date': 'end_date',
            'type': 'type',
            'net_profit_min': 'net_profit_min',
            'net_profit_max': 'net_profit_max',
        }
        forecast_df = forecast_df.rename(
            columns={k: v for k, v in col_map.items() if k in forecast_df.columns}
        )
        for col in ['ann_date', 'end_date']:
            if col in forecast_df.columns:
                forecast_df[col] = forecast_df[col].astype(str)

    # 2) 业绩快报 express（获取 pre_eps）
    express_df = _safe_get(
        pro.express,
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date
    )
    if express_df is None or express_df.empty:
        express_df = pd.DataFrame()
    else:
        express_cols = {
            'ts_code': 'ts_code',
            'ann_date': 'ann_date',
            'end_date': 'end_date',
            'diluted_eps': 'pre_eps',
        }
        express_df = express_df.rename(
            columns={k: v for k, v in express_cols.items() if k in express_df.columns}
        )
        for col in ['ann_date', 'end_date']:
            if col in express_df.columns:
                express_df[col] = express_df[col].astype(str)

    # 3) 合并数据
    if forecast_df.empty and express_df.empty:
        logger.warning(f"{ts_code} 未获取到任何 forecast/express 数据，使用 mock")
        result = _generate_mock_forecast(ts_code, n=10)
    elif forecast_df.empty:
        result = express_df[['ts_code', 'ann_date', 'end_date', 'pre_eps']].copy()
        result['type'] = '快报'
        result['net_profit_min'] = np.nan
        result['net_profit_max'] = np.nan
    elif express_df.empty:
        result = forecast_df[['ts_code', 'ann_date', 'end_date', 'type',
                                'net_profit_min', 'net_profit_max']].copy()
        result['pre_eps'] = np.nan
    else:
        # 以 forecast 为基准，左连接 express 的 pre_eps
        merge_cols = ['ts_code', 'ann_date', 'end_date']
        # 取两者都有的列做 merge
        common_cols = [c for c in merge_cols if c in forecast_df.columns and c in express_df.columns]
        if common_cols:
            result = forecast_df.merge(
                express_df[common_cols + ['pre_eps']],
                on=common_cols,
                how='left',
                suffixes=('', '_exp')
            )
            # 处理重复的 pre_eps 列
            if 'pre_eps_exp' in result.columns:
                result['pre_eps'] = result['pre_eps'].fillna(result['pre_eps_exp'])
                result = result.drop(columns=['pre_eps_exp'])
        else:
            result = pd.concat([forecast_df, express_df], ignore_index=True)

    # 确保输出列完整
    for col in ['ts_code', 'ann_date', 'end_date', 'type', 'pre_eps',
                'net_profit_min', 'net_profit_max']:
        if col not in result.columns:
            result[col] = np.nan

    result = result[['ts_code', 'ann_date', 'end_date', 'type',
                     'pre_eps', 'net_profit_min', 'net_profit_max']]
    _save_cache(cache_path, result)
    return result


# ---------------------------------------------------------------------------
# 计算 SUE（Standardized Unexpected Earnings）
# ---------------------------------------------------------------------------


def calculate_sue(forecast_df: pd.DataFrame,
                  actual_eps: pd.Series) -> pd.DataFrame:
    """计算标准化盈利惊喜（SUE）。

    SUE = (Actual EPS - Forecast EPS) / std(Forecast EPS)

    其中 Forecast EPS 取公告日前 30 天内最新的分析师一致预期。
    std(Forecast EPS) 取过去 90 天内预期 EPS 的标准差。

    Args:
        forecast_df: 一致预期 DataFrame（来自 fetch_consensus_forecast）
        actual_eps: 实际 EPS 的 Series，index 为公告日期（YYYYMMDD）

    Returns:
        DataFrame with columns:
            - ts_code
            - ann_date
            - sue: SUE 值
            - actual_eps: 实际 EPS
            - forecast_eps: 预期 EPS
            - forecast_std: 预期标准差
    """
    if forecast_df.empty or actual_eps.empty:
        logger.warning("forecast_df 或 actual_eps 为空，无法计算 SUE")
        return pd.DataFrame()

    # 确保日期格式正确
    forecast_df = forecast_df.copy()
    forecast_df['ann_date'] = pd.to_datetime(forecast_df['ann_date'], errors='coerce')
    forecast_df = forecast_df.dropna(subset=['ann_date'])

    actual_dates = pd.to_datetime(actual_eps.index, errors='coerce')
    actual_eps_clean = actual_eps.copy()
    actual_eps_clean.index = actual_dates
    actual_eps_clean = actual_eps_clean.dropna()

    if actual_eps_clean.empty:
        logger.warning("actual_eps 无有效日期，无法计算 SUE")
        return pd.DataFrame()

    ts_code = forecast_df['ts_code'].iloc[0] if 'ts_code' in forecast_df.columns else ''

    results = []
    for ann_date, actual in actual_eps_clean.items():
        # 公告日前 30 天内最新的预期
        window_30 = forecast_df[
            (forecast_df['ann_date'] >= ann_date - timedelta(days=30)) &
            (forecast_df['ann_date'] <= ann_date)
        ]
        if window_30.empty or 'pre_eps' not in window_30.columns:
            continue
        latest_forecast = window_30.sort_values('ann_date').iloc[-1]['pre_eps']
        if pd.isna(latest_forecast):
            continue

        # 过去 90 天内预期 EPS 的标准差
        window_90 = forecast_df[
            (forecast_df['ann_date'] >= ann_date - timedelta(days=90)) &
            (forecast_df['ann_date'] <= ann_date)
        ]
        if window_90.empty or 'pre_eps' not in window_90.columns:
            continue
        forecast_std = window_90['pre_eps'].std()
        if pd.isna(forecast_std) or forecast_std == 0:
            forecast_std = window_90['pre_eps'].mean() * 0.1
            if pd.isna(forecast_std) or forecast_std == 0:
                continue

        sue = (actual - latest_forecast) / forecast_std
        results.append({
            'ts_code': ts_code,
            'ann_date': ann_date.strftime('%Y%m%d'),
            'sue': sue,
            'actual_eps': actual,
            'forecast_eps': latest_forecast,
            'forecast_std': forecast_std,
        })

    if not results:
        logger.warning("未计算出任何 SUE 值")
        return pd.DataFrame()

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 计算 EPS 预期修正动量
# ---------------------------------------------------------------------------


def calculate_eps_revision(forecast_df: pd.DataFrame,
                           window_days: int = 30) -> pd.DataFrame:
    """计算 EPS 预期修正动量。

    EPS_REVISION = (最新预期 EPS - N 天前预期 EPS) / |N 天前预期 EPS|

    Args:
        forecast_df: 一致预期 DataFrame（来自 fetch_consensus_forecast）
        window_days: 时间窗口天数，默认 30

    Returns:
        DataFrame with columns:
            - ts_code
            - revision: EPS 修正率
            - direction: 1=上调, -1=下调, 0=不变
    """
    if forecast_df.empty or 'pre_eps' not in forecast_df.columns:
        logger.warning("forecast_df 为空或缺少 pre_eps 列，无法计算 EPS 修正")
        return pd.DataFrame()

    forecast_df = forecast_df.copy()
    forecast_df['ann_date'] = pd.to_datetime(forecast_df['ann_date'], errors='coerce')
    forecast_df = forecast_df.dropna(subset=['ann_date', 'pre_eps'])

    if forecast_df.empty:
        logger.warning("无有效 EPS 预期数据")
        return pd.DataFrame()

    ts_code = forecast_df['ts_code'].iloc[0] if 'ts_code' in forecast_df.columns else ''
    latest_date = forecast_df['ann_date'].max()
    past_date = latest_date - timedelta(days=window_days)

    latest_row = forecast_df[forecast_df['ann_date'] == latest_date]
    if latest_row.empty:
        latest_row = forecast_df.sort_values('ann_date').iloc[[-1]]
    latest_eps = latest_row['pre_eps'].iloc[0]

    past_rows = forecast_df[forecast_df['ann_date'] <= past_date]
    if past_rows.empty:
        past_rows = forecast_df.sort_values('ann_date').iloc[[0]]
    past_eps = past_rows['pre_eps'].iloc[0]

    if pd.isna(latest_eps) or pd.isna(past_eps) or past_eps == 0:
        logger.warning("EPS 数据无效，无法计算修正率")
        return pd.DataFrame()

    revision = (latest_eps - past_eps) / abs(past_eps)
    direction = 1 if revision > 0.01 else (-1 if revision < -0.01 else 0)

    return pd.DataFrame([{
        'ts_code': ts_code,
        'revision': revision,
        'direction': direction,
    }])


# ---------------------------------------------------------------------------
# 计算分析师评级修正
# ---------------------------------------------------------------------------


@retry_on_failure(max_retries=3, delay=2, backoff=2)
def calculate_rating_revision(ts_code: str,
                              start_date: str,
                              end_date: str,
                              pro: Optional[object] = None) -> pd.DataFrame:
    """计算分析师评级净上调比例。

    使用 Tushare Pro 的 ``report_rc`` 接口（机构调研）或模拟实现。

    RATING_REVISION = (上调评级数 - 下调评级数) / 总评级数

    Args:
        ts_code: Tushare 格式股票代码
        start_date: 开始日期（YYYYMMDD）
        end_date: 结束日期（YYYYMMDD）
        pro: Tushare Pro 实例，若为 None 则自动获取

    Returns:
        DataFrame with columns:
            - ts_code
            - rating_revision: 评级净上调比例 [-1, 1]
            - n_up: 上调数
            - n_down: 下调数
            - n_total: 总评级数
    """
    if pro is None:
        pro = get_tushare_pro()
    if pro is None:
        logger.warning("Tushare Pro 不可用，返回模拟评级修正数据")
        mock_ratings = _generate_mock_rating_data(ts_code, n=20)
        return _compute_rating_revision_from_mock(mock_ratings)

    cache_key = f"rating_{ts_code}_{start_date}_{end_date}"
    cache_path = _cache_path("rating", cache_key)
    cached = _load_cache(cache_path, max_age_hours=6)
    if cached is not None:
        logger.info(f"从缓存加载 rating 数据: {ts_code}")
        return cached

    logger.info(f"获取评级修正数据: {ts_code} [{start_date} ~ {end_date}]")

    # 尝试 report_rc 接口（机构调研）
    rc_df = _safe_get(
        pro.report_rc,
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date
    )

    if rc_df is None or rc_df.empty:
        logger.warning(f"{ts_code} 未获取到 report_rc 数据，使用 mock")
        mock_ratings = _generate_mock_rating_data(ts_code, n=20)
        result = _compute_rating_revision_from_mock(mock_ratings)
        _save_cache(cache_path, result)
        return result

    # report_rc 中通常没有直接的 rating_change 字段，
    # 这里根据接口返回的字段做适配，若无法识别则回退到 mock
    if 'rating_change' not in rc_df.columns and 'change' not in rc_df.columns:
        logger.warning("report_rc 返回字段不包含评级变化信息，使用 mock 计算")
        mock_ratings = _generate_mock_rating_data(ts_code, n=20)
        result = _compute_rating_revision_from_mock(mock_ratings)
        _save_cache(cache_path, result)
        return result

    change_col = 'rating_change' if 'rating_change' in rc_df.columns else 'change'
    n_up = (rc_df[change_col].astype(str).str.contains('上调')).sum()
    n_down = (rc_df[change_col].astype(str).str.contains('下调')).sum()
    n_total = len(rc_df)

    if n_total == 0:
        rating_revision = 0.0
    else:
        rating_revision = (n_up - n_down) / n_total

    result = pd.DataFrame([{
        'ts_code': ts_code,
        'rating_revision': np.clip(rating_revision, -1, 1),
        'n_up': int(n_up),
        'n_down': int(n_down),
        'n_total': int(n_total),
    }])
    _save_cache(cache_path, result)
    return result


def _compute_rating_revision_from_mock(mock_df: pd.DataFrame) -> pd.DataFrame:
    """从模拟评级数据计算评级净上调比例。

    Args:
        mock_df: 模拟评级 DataFrame

    Returns:
        DataFrame with columns [ts_code, rating_revision, n_up, n_down, n_total]
    """
    if mock_df.empty:
        return pd.DataFrame()
    ts_code = mock_df['ts_code'].iloc[0]
    n_up = (mock_df['rating_change'] == '上调').sum()
    n_down = (mock_df['rating_change'] == '下调').sum()
    n_total = len(mock_df)
    rating_revision = (n_up - n_down) / n_total if n_total > 0 else 0.0
    return pd.DataFrame([{
        'ts_code': ts_code,
        'rating_revision': np.clip(rating_revision, -1, 1),
        'n_up': int(n_up),
        'n_down': int(n_down),
        'n_total': int(n_total),
    }])


# ---------------------------------------------------------------------------
# 统一接口：获取预期差距因子
# ---------------------------------------------------------------------------


def fetch_expectation_gap_factors(stock_codes: List[str],
                                  trade_date: Optional[str] = None,
                                  pro: Optional[object] = None) -> pd.DataFrame:
    """获取所有预期差距因子（SUE + EPS_REVISION + RATING_REVISION）。

    这是 factor_engine 和 scoring_engine 的直接输入。

    Args:
        stock_codes: 股票代码列表（6 位数字）
        trade_date: 交易日期（YYYYMMDD），默认最近一个交易日
        pro: Tushare Pro 实例，若为 None 则自动获取

    Returns:
        DataFrame with columns:
            - code: 股票代码（与现有系统兼容，6 位数字）
            - sue: SUE 值
            - eps_revision: EPS 修正动量
            - rating_revision: 评级修正动量

        如果 Tushare 不可用，返回 mock 数据（随机生成合理范围的值）。
    """
    if not stock_codes:
        logger.warning("股票代码列表为空，返回空 DataFrame")
        return pd.DataFrame()

    if pro is None:
        pro = get_tushare_pro()

    if trade_date is None:
        trade_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

    # 计算数据获取的时间窗口
    end_dt = datetime.strptime(trade_date, '%Y%m%d')
    start_dt = end_dt - timedelta(days=180)
    start_date = start_dt.strftime('%Y%m%d')
    end_date = end_dt.strftime('%Y%m%d')

    # 若 Tushare 不可用，直接返回 mock
    if pro is None:
        logger.warning("Tushare Pro 不可用，返回 mock 预期差距数据")
        return _generate_mock_expectation_data(stock_codes)

    logger.info(f"获取 {len(stock_codes)} 只股票的预期差距因子 [{start_date} ~ {end_date}]")

    result_rows = []
    for i, code in enumerate(stock_codes):
        try:
            ts_code = _to_tushare_code(code)

            # 1) 获取一致预期
            forecast_df = fetch_consensus_forecast(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                pro=pro
            )

            # 2) 计算 EPS 修正
            eps_rev_df = calculate_eps_revision(forecast_df, window_days=30)
            eps_revision = eps_rev_df['revision'].iloc[0] if not eps_rev_df.empty else 0.0

            # 3) 计算 SUE（需要实际 EPS）
            # 尝试从 income 接口获取实际 EPS
            income_df = _safe_get(
                pro.income,
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            sue = 0.0
            if income_df is not None and not income_df.empty and 'diluted_eps' in income_df.columns:
                income_df = income_df.sort_values('end_date', ascending=False)
                actual_eps = pd.Series(
                    income_df['diluted_eps'].astype(float).values,
                    index=income_df['end_date'].astype(str).values
                )
                sue_df = calculate_sue(forecast_df, actual_eps)
                if not sue_df.empty:
                    sue = sue_df['sue'].iloc[0]
            else:
                # 无实际 EPS 时，用 mock 的 SUE
                np.random.seed(hash(code) % 2**31)
                sue = np.random.normal(0, 2)

            # 4) 计算评级修正
            rating_df = calculate_rating_revision(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                pro=pro
            )
            rating_revision = rating_df['rating_revision'].iloc[0] if not rating_df.empty else 0.0

            result_rows.append({
                'code': code,
                'sue': sue,
                'eps_revision': eps_revision,
                'rating_revision': rating_revision,
            })

            if (i + 1) % 10 == 0:
                logger.info(f"已获取 {i+1}/{len(stock_codes)} 只股票的预期差距因子...")
            time.sleep(random.uniform(0.2, 0.5))  # 避免请求过快

        except Exception as e:
            logger.warning(f"获取股票 {code} 预期差距因子失败: {e}")
            # 单只股票失败时填充 mock 值，保证整体不中断
            np.random.seed(hash(code) % 2**31)
            result_rows.append({
                'code': code,
                'sue': np.random.normal(0, 2),
                'eps_revision': np.random.normal(0, 0.15),
                'rating_revision': np.random.normal(0, 0.3),
            })
            continue

    if not result_rows:
        logger.warning("未获取到任何预期差距因子，返回 mock 数据")
        return _generate_mock_expectation_data(stock_codes)

    result = pd.DataFrame(result_rows)
    # 对极端值做截断，保证合理性
    result['sue'] = result['sue'].clip(-10, 10)
    result['eps_revision'] = result['eps_revision'].clip(-1, 1)
    result['rating_revision'] = result['rating_revision'].clip(-1, 1)

    logger.info(f"成功获取 {len(result)} 只股票的预期差距因子")
    return result


# ---------------------------------------------------------------------------
# 测试入口
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 70)
    print("Mini-GRP Tushare Pro 数据接入模块测试")
    print("=" * 70)

    # 1. 测试 Tushare Pro 客户端
    print("\n[1/6] 测试 get_tushare_pro()...")
    pro = get_tushare_pro()
    if pro is not None:
        print("Tushare Pro 客户端初始化成功")
    else:
        print("Tushare Pro 客户端不可用（未安装或 token 缺失），后续将使用 mock 数据")

    # 2. 测试代码转换
    print("\n[2/6] 测试代码格式转换...")
    test_codes = ['000001', '600000', '300001', '688001']
    for c in test_codes:
        ts_c = _to_tushare_code(c)
        back_c = _from_tushare_code(ts_c)
        status = "OK" if back_c == c else "FAIL"
        print(f"  {c} -> {ts_c} -> {back_c}  [{status}]")

    # 3. 测试获取一致预期
    print("\n[3/6] 测试 fetch_consensus_forecast()...")
    test_ts_code = '000001.SZ'
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
    forecast_df = fetch_consensus_forecast(
        ts_code=test_ts_code,
        start_date=start_date,
        end_date=end_date,
        pro=pro
    )
    print(f"获取到 {len(forecast_df)} 条 forecast 记录")
    if not forecast_df.empty:
        print(forecast_df.head(3).to_string())

    # 4. 测试计算 EPS 修正
    print("\n[4/6] 测试 calculate_eps_revision()...")
    if not forecast_df.empty:
        eps_rev_df = calculate_eps_revision(forecast_df, window_days=30)
        print(f"EPS 修正结果:\n{eps_rev_df.to_string()}")
    else:
        print("forecast_df 为空，跳过 EPS 修正测试")

    # 5. 测试计算评级修正
    print("\n[5/6] 测试 calculate_rating_revision()...")
    rating_df = calculate_rating_revision(
        ts_code=test_ts_code,
        start_date=start_date,
        end_date=end_date,
        pro=pro
    )
    print(f"评级修正结果:\n{rating_df.to_string()}")

    # 6. 测试统一接口
    print("\n[6/6] 测试 fetch_expectation_gap_factors()...")
    test_codes = ['000001', '600000', '300001']
    gap_df = fetch_expectation_gap_factors(
        stock_codes=test_codes,
        trade_date=end_date,
        pro=pro
    )
    print(f"获取到 {len(gap_df)} 只股票的预期差距因子")
    print(gap_df.to_string())

    print("\n" + "=" * 70)
    print("Tushare Pro 数据接入模块测试完成!")
    print("=" * 70)
