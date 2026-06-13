#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini-GRP 因子计算模块
对原始数据进行去极值、标准化和行业中性化处理，计算19个因子的标准化值

作者: Quant Dev
日期: 2025-06-09
"""

import logging
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 因子定义
# ---------------------------------------------------------------------------

# 因子类别、名称、列名、方向（1=正向，-1=负向）
FACTOR_DEFINITIONS = [
    # 价值因子 (Value) - 方向: -1 (越低越好)
    {'category': 'value', 'name': '市盈率_TTM', 'column': 'pe_ttm', 'direction': -1},
    {'category': 'value', 'name': '市净率_LF', 'column': 'pb_lf', 'direction': -1},
    {'category': 'value', 'name': '市销率_TTM', 'column': 'ps_ttm', 'direction': -1},
    {'category': 'value', 'name': 'EV_EBITDA', 'column': 'ev_ebitda', 'direction': -1},
    {'category': 'value', 'name': '股息率', 'column': 'dividend_yield', 'direction': 1},

    # 质量因子 (Quality) - 方向: 1 (越高越好), debt_to_equity 为 -1
    {'category': 'quality', 'name': 'ROE_扣非', 'column': 'roe_deducted', 'direction': 1},
    {'category': 'quality', 'name': 'ROA', 'column': 'roa', 'direction': 1},
    {'category': 'quality', 'name': '毛利率', 'column': 'gross_margin', 'direction': 1},
    {'category': 'quality', 'name': '净利率', 'column': 'net_margin', 'direction': 1},
    {'category': 'quality', 'name': '负债权益比', 'column': 'debt_to_equity', 'direction': -1},

    # 增长因子 (Growth) - 方向: 1 (越高越好)
    {'category': 'growth', 'name': '营收增长率', 'column': 'revenue_yoy', 'direction': 1},
    {'category': 'growth', 'name': '盈利增长率', 'column': 'profit_yoy', 'direction': 1},
    {'category': 'growth', 'name': 'FCF收益率', 'column': 'fcf_yield', 'direction': 1},

    # 动量因子 (Momentum) - 方向: 1 (越高越好)
    {'category': 'momentum', 'name': '1个月收益', 'column': 'return_1m', 'direction': 1},
    {'category': 'momentum', 'name': '3个月收益', 'column': 'return_3m', 'direction': 1},
    {'category': 'momentum', 'name': '12个月收益', 'column': 'return_12m', 'direction': 1},

    # 预期差距因子 (Expectation Gap) - 方向: 1 (越高越好)
    {'category': 'expectation', 'name': 'SUE', 'column': 'sue', 'direction': 1},
    {'category': 'expectation', 'name': 'EPS_REVISION', 'column': 'eps_revision', 'direction': 1},
    {'category': 'expectation', 'name': 'RATING_REVISION', 'column': 'rating_revision', 'direction': 1},
]

# 按类别分组
FACTOR_CATEGORIES = {
    'value': [f['column'] for f in FACTOR_DEFINITIONS if f['category'] == 'value'],
    'quality': [f['column'] for f in FACTOR_DEFINITIONS if f['category'] == 'quality'],
    'growth': [f['column'] for f in FACTOR_DEFINITIONS if f['category'] == 'growth'],
    'momentum': [f['column'] for f in FACTOR_DEFINITIONS if f['category'] == 'momentum'],
    'expectation': [f['column'] for f in FACTOR_DEFINITIONS if f['category'] == 'expectation'],
}

# 所有因子列名
ALL_FACTOR_COLUMNS = [f['column'] for f in FACTOR_DEFINITIONS]

# 因子方向映射
FACTOR_DIRECTIONS = {f['column']: f['direction'] for f in FACTOR_DEFINITIONS}


# ---------------------------------------------------------------------------
# 核心数据处理函数
# ---------------------------------------------------------------------------

def winsorize(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    """
    去极值（缩尾处理）

    用上下分位数替代极端值，防止异常值对因子计算产生过大影响。
    默认使用 5% 和 95% 分位数。

    Args:
        series: 输入数据序列
        lower: 下分位比例，默认 0.05
        upper: 上分位比例，默认 0.95

    Returns:
        去极值后的序列
    """
    if series.empty or series.isna().all():
        return series

    # 只考虑非空值计算分位数
    valid = series.dropna()
    if len(valid) == 0:
        return series

    lower_bound = valid.quantile(lower)
    upper_bound = valid.quantile(upper)

    # 缩尾处理
    result = series.clip(lower=lower_bound, upper=upper_bound)
    return result


def winsorize_mad(series: pd.Series, n_mad: float = 3.0) -> pd.Series:
    """Clip a series around its median using a scaled median absolute deviation."""
    if series.empty or series.isna().all():
        return series
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return series
    median = float(valid.median())
    mad = float((valid - median).abs().median())
    if mad <= 0 or not np.isfinite(mad):
        return series
    robust_sigma = 1.4826 * mad
    return series.clip(lower=median - n_mad * robust_sigma, upper=median + n_mad * robust_sigma)


def standardize(series: pd.Series) -> pd.Series:
    """
    标准化（Z-Score）

    将数据转换为均值为0、标准差为1的标准正态分布
    公式: z = (x - mean) / std

    Args:
        series: 输入数据序列

    Returns:
        标准化后的序列
    """
    if series.empty or series.isna().all():
        return series

    valid = series.dropna()
    if len(valid) == 0:
        return series

    mean = valid.mean()
    std = valid.std()

    if std == 0 or np.isnan(std):
        # 标准差为0，返回全0序列
        return pd.Series(0, index=series.index)

    result = (series - mean) / std
    return result


def neutralize_industry(factor: pd.Series, industry: pd.Series) -> pd.Series:
    """
    行业中性化

    每个值减去其所在行业的均值，消除行业间的系统性差异。
    适用于跨行业比较因子值。

    Args:
        factor: 因子值序列
        industry: 行业分类序列（与 factor 同索引）

    Returns:
        行业中性化后的因子序列
    """
    if factor.empty or factor.isna().all():
        return factor

    if industry.empty:
        logger.warning("行业数据为空，跳过行业中性化")
        return factor

    # 构建 DataFrame 方便分组计算
    df = pd.DataFrame({
        'factor': factor,
        'industry': industry
    })

    # 按行业计算均值并减去
    industry_mean = df.groupby('industry')['factor'].transform('mean')

    result = factor - industry_mean
    return result


def neutralize_market_cap(factor: pd.Series, market_cap: pd.Series) -> pd.Series:
    """Remove linear log-market-cap exposure and return cross-sectional residuals."""
    values = pd.DataFrame(
        {
            "factor": pd.to_numeric(factor, errors="coerce"),
            "market_cap": pd.to_numeric(market_cap, errors="coerce"),
        }
    )
    valid = values["factor"].notna() & values["market_cap"].gt(0)
    if valid.sum() < 10 or values.loc[valid, "market_cap"].nunique() < 3:
        return factor
    x = np.log(values.loc[valid, "market_cap"].to_numpy(dtype=float))
    design = np.column_stack([np.ones(len(x)), x])
    y = values.loc[valid, "factor"].to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = pd.Series(np.nan, index=factor.index, dtype=float)
    residual.loc[valid] = y - design @ coefficients
    return residual


# ---------------------------------------------------------------------------
# 缺失值处理
# ---------------------------------------------------------------------------

def fill_missing_by_industry(df: pd.DataFrame,
                              factor_cols: List[str],
                              industry_col: str = 'sw_industry_name') -> pd.DataFrame:
    """
    使用行业均值填充缺失值

    对于每个因子列，按行业计算均值，然后用行业均值填充该行业的缺失值。
    如果整个行业的值都缺失，则使用全市场均值填充。

    Args:
        df: 包含因子列和行业列的 DataFrame
        factor_cols: 需要填充的因子列名列表
        industry_col: 行业列名

    Returns:
        填充后的 DataFrame
    """
    result = df.copy()

    if industry_col not in result.columns:
        logger.warning(f"行业列 '{industry_col}' 不存在，使用全局均值填充")
        for col in factor_cols:
            if col in result.columns:
                result[col] = result[col].fillna(result[col].mean())
        return result

    for col in factor_cols:
        if col not in result.columns:
            continue

        # 按行业均值填充
        industry_means = result.groupby(industry_col)[col].transform('mean')
        result[col] = result[col].fillna(industry_means)

        # 如果仍有缺失（整个行业都缺失），用全局均值填充
        global_mean = result[col].mean()
        result[col] = result[col].fillna(global_mean)

        # 如果还是缺失（比如全局均值也是NaN），填0
        result[col] = result[col].fillna(0)

    return result


# ---------------------------------------------------------------------------
# 主因子计算函数
# ---------------------------------------------------------------------------

def calculate_factors(raw_data: pd.DataFrame,
                      apply_winsorize: bool = True,
                      apply_standardize: bool = True,
                      apply_industry_neutral: bool = False,
                      winsorize_method: str = "quantile",
                      apply_size_neutral: bool = False,
                      market_cap_col: str = "market_cap") -> pd.DataFrame:
    """
    计算所有因子的标准化值

    处理流程:
        1. 检查输入数据和必要的列
        2. 填充缺失值（行业均值填充）
        3. 对每个因子进行去极值 (winsorize)
        4. 对每个因子进行标准化 (standardize)
        5. 根据因子方向调整符号（负向因子取反）
        6. （可选）进行行业中性化
        7. 保留至少50%的股票

    Args:
        raw_data: data_engine 输出的原始数据，包含原始因子列
        apply_winsorize: 是否进行去极值处理，默认 True
        apply_standardize: 是否进行标准化处理，默认 True
        apply_industry_neutral: 是否进行行业中性化，默认 False
        winsorize_method: 去极值方法，支持 quantile 或 mad
        apply_size_neutral: 是否使用对数市值 OLS 残差做市值中性化
        market_cap_col: 当期市值字段名

    Returns:
        DataFrame: 原始数据 + 各因子标准化列（列名为 {column}_z）
                   以及各因子类别得分（value_score, quality_score 等）
    """
    logger.info("开始计算因子...")

    if raw_data is None or raw_data.empty:
        logger.error("输入数据为空")
        return pd.DataFrame()

    result = raw_data.copy()
    n_total = len(result)
    if winsorize_method not in {"quantile", "mad"}:
        raise ValueError("winsorize_method must be 'quantile' or 'mad'")

    # ------------------------------------------------------------------
    # 1. 检查可用的因子列
    # ------------------------------------------------------------------
    available_factors = []
    unusable_factors = []
    for col in ALL_FACTOR_COLUMNS:
        if col not in result.columns:
            continue
        numeric = pd.to_numeric(result[col], errors='coerce')
        result[col] = numeric
        if numeric.notna().sum() >= 2 and numeric.nunique(dropna=True) > 1:
            available_factors.append(col)
        else:
            unusable_factors.append(col)
    missing_factors = [col for col in ALL_FACTOR_COLUMNS if col not in result.columns]

    logger.info(f"可用因子: {available_factors}")
    if missing_factors:
        logger.warning(f"缺失因子（将跳过）: {missing_factors}")
    if unusable_factors:
        logger.warning(f"无有效横截面差异的因子（将跳过）: {unusable_factors}")

    if not available_factors:
        logger.error("没有可用的因子列，无法计算")
        return result

    # ------------------------------------------------------------------
    # 2. 缺失值处理
    # ------------------------------------------------------------------
    industry_col = 'sw_industry_name' if 'sw_industry_name' in result.columns else None
    market_col = 'market' if 'market' in result.columns and result['market'].nunique() > 1 else None

    # 检查缺失比例
    for col in available_factors:
        missing_ratio = result[col].isna().mean()
        if missing_ratio > 0.5:
            logger.warning(f"因子 {col} 缺失率 {missing_ratio:.1%}，超过50%")

    # 使用行业均值填充缺失值
    group_cols = [col for col in (market_col, industry_col) if col]
    for col in available_factors:
        if group_cols:
            group_means = result.groupby(group_cols, dropna=False)[col].transform('mean')
            result[col] = result[col].fillna(group_means)
        if market_col:
            market_means = result.groupby(market_col, dropna=False)[col].transform('mean')
            result[col] = result[col].fillna(market_means)
        result[col] = result[col].fillna(result[col].mean()).fillna(0)

    # ------------------------------------------------------------------
    # 3. 逐个因子处理：去极值 -> 标准化 -> 方向调整
    # ------------------------------------------------------------------
    factor_z_scores = {}  # 存储标准化后的因子值

    for factor_col in available_factors:
        try:
            direction = FACTOR_DIRECTIONS.get(factor_col, 1)

            def process_cross_section(values: pd.Series) -> pd.Series:
                if not apply_winsorize:
                    processed = values
                elif winsorize_method == "mad":
                    processed = winsorize_mad(values)
                else:
                    processed = winsorize(values)
                return standardize(processed) if apply_standardize else processed

            if market_col:
                values = result.groupby(market_col, group_keys=False)[factor_col].transform(
                    process_cross_section
                )
            else:
                values = process_cross_section(result[factor_col].copy())

            # 步骤 3: 方向调整（负向因子取反，使得越大越好）
            values = values * direction

            if apply_size_neutral and market_cap_col in result.columns:
                values = neutralize_market_cap(values, result[market_cap_col])
                values = standardize(values) if apply_standardize else values

            # 步骤 4: （可选）行业中性化
            if apply_industry_neutral and industry_col and industry_col in result.columns:
                industry_key = result[industry_col].astype(str)
                if market_col:
                    industry_key = result[market_col].astype(str) + "::" + industry_key
                values = neutralize_industry(values, industry_key)
                values = (
                    values.groupby(result[market_col]).transform(standardize)
                    if market_col
                    else standardize(values)
                )

            factor_z_scores[factor_col] = values

        except Exception as e:
            logger.error(f"处理因子 {factor_col} 时出错: {e}")
            factor_z_scores[factor_col] = pd.Series(np.nan, index=result.index)

    # 将标准化因子添加到结果中
    for col, values in factor_z_scores.items():
        result[f'{col}_z'] = values

    # ------------------------------------------------------------------
    # 4. 计算各类别综合得分
    # ------------------------------------------------------------------
    z_score_cols = [f'{col}_z' for col in available_factors]

    for category, factor_list in FACTOR_CATEGORIES.items():
        category_z_cols = [f'{col}_z' for col in factor_list if f'{col}_z' in result.columns]
        if category_z_cols:
            # 类别得分为该类别所有因子Z值的均值
            result[f'{category}_score'] = result[category_z_cols].mean(axis=1)
            logger.info(f"计算 {category} 得分，使用因子: {category_z_cols}")
        else:
            result[f'{category}_score'] = np.nan
            logger.warning(f"类别 {category} 没有可用因子")

    # ------------------------------------------------------------------
    # 5. 计算综合因子得分（所有可用因子的Z值均值）
    # ------------------------------------------------------------------
    available_z_cols = [col for col in z_score_cols if col in result.columns]
    if available_z_cols:
        result['composite_factor_score'] = result[available_z_cols].mean(axis=1)

    # ------------------------------------------------------------------
    # 6. 质量控制：删除因子值全缺失的股票
    # ------------------------------------------------------------------
    valid_mask = result[available_z_cols].notna().any(axis=1)
    n_valid = valid_mask.sum()

    if n_valid < n_total * 0.5:
        logger.warning(
            f"有效股票数量 {n_valid} 不足总数 {n_total} 的50%，"
            f"保留所有股票"
        )
    else:
        result = result[valid_mask].reset_index(drop=True)
        logger.info(f"删除 {n_total - n_valid} 只因子全缺失的股票")

    logger.info(
        f"因子计算完成: {len(result)} 只股票, "
        f"{len(available_factors)} 个因子, "
        f"{len(available_z_cols)} 个标准化因子"
    )

    return result


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def get_factor_summary(factor_df: pd.DataFrame) -> pd.DataFrame:
    """
    获取因子统计摘要

    Args:
        factor_df: calculate_factors 的输出

    Returns:
        因子统计摘要 DataFrame
    """
    z_cols = [col for col in factor_df.columns if col.endswith('_z')]

    summary = []
    for col in z_cols:
        original_col = col[:-2]  # 去掉 '_z'
        direction = FACTOR_DIRECTIONS.get(original_col, 1)
        category = None
        for cat, factors in FACTOR_CATEGORIES.items():
            if original_col in factors:
                category = cat
                break

        summary.append({
            'factor': original_col,
            'category': category,
            'direction': direction,
            'mean': factor_df[col].mean(),
            'std': factor_df[col].std(),
            'min': factor_df[col].min(),
            'max': factor_df[col].max(),
            'median': factor_df[col].median(),
            'missing': factor_df[col].isna().sum(),
        })

    return pd.DataFrame(summary)


def get_available_factors(df: pd.DataFrame) -> List[str]:
    """
    获取数据框中可用的原始因子列

    Args:
        df: 原始数据 DataFrame

    Returns:
        可用的因子列名列表
    """
    return [col for col in ALL_FACTOR_COLUMNS if col in df.columns]


# ---------------------------------------------------------------------------
# 测试入口
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Mini-GRP 因子计算模块测试")
    print("=" * 60)

    # 生成模拟原始数据
    print("\n[1/5] 生成模拟原始数据...")
    np.random.seed(42)
    n = 100

    industries = ['银行', '医药', '电子', '食品', '电力', '计算机', '汽车', '化工']

    raw_data = pd.DataFrame({
        'code': [f"{600000 + i:06d}" for i in range(n)],
        'name': [f"股票_{i}" for i in range(n)],
        'pe_ttm': np.random.lognormal(3, 0.5, n),
        'pb_lf': np.random.lognormal(0.5, 0.5, n),
        'ps_ttm': np.random.lognormal(1, 0.6, n),
        'ev_ebitda': np.random.lognormal(2.5, 0.6, n),
        'dividend_yield': np.random.exponential(2, n),
        'roe_deducted': np.random.normal(10, 8, n),
        'roa': np.random.normal(5, 4, n),
        'gross_margin': np.random.normal(30, 15, n),
        'net_margin': np.random.normal(12, 10, n),
        'debt_to_equity': np.random.lognormal(4, 0.5, n),
        'revenue_yoy': np.random.normal(15, 20, n),
        'profit_yoy': np.random.normal(12, 25, n),
        'fcf_yield': np.random.normal(3, 2, n),
        'return_1m': np.random.normal(0, 8, n),
        'return_3m': np.random.normal(0, 15, n),
        'return_12m': np.random.normal(10, 35, n),
        'sue': np.random.normal(0, 2, n),
        'eps_revision': np.random.normal(0, 5, n),
        'rating_revision': np.random.normal(0, 3, n),
        'sw_industry_name': np.random.choice(industries, n),
    })
    print(f"生成 {len(raw_data)} 行模拟数据")
    print(raw_data.head(3)[['code', 'pe_ttm', 'pb_lf', 'roe_deducted', 'sue', 'eps_revision']].to_string())

    # 2. 测试因子计算
    print("\n[2/5] 测试 calculate_factors()...")
    factor_df = calculate_factors(raw_data)
    print(f"因子计算完成: {len(factor_df)} 行")

    z_cols = [c for c in factor_df.columns if c.endswith('_z')]
    print(f"标准化因子列: {z_cols}")

    # 3. 测试因子摘要
    print("\n[3/5] 因子统计摘要:")
    summary = get_factor_summary(factor_df)
    print(summary[['factor', 'category', 'mean', 'std', 'min', 'max']].to_string())

    # 4. 验证因子方向
    print("\n[4/5] 验证因子方向（高值=好）:")
    for col, direction in [('pe_ttm', -1), ('pb_lf', -1), ('roe_deducted', 1), ('return_1m', 1), ('sue', 1), ('eps_revision', 1)]:
        z_col = f'{col}_z'
        corr = raw_data[col].corr(factor_df[z_col])
        expected = direction  # 标准化后乘以方向，所以原始值与Z值的相关性应该与方向一致
        print(f"  {col}: 原始值与Z值相关性 = {corr:.3f} (期望方向: {'正' if direction > 0 else '负'})")

    # 5. 检查得分列
    print("\n[5/5] 类别得分:")
    score_cols = ['value_score', 'quality_score', 'growth_score', 'momentum_score', 'expectation_score']
    for sc in score_cols:
        if sc in factor_df.columns:
            print(f"  {sc}: mean={factor_df[sc].mean():.4f}, std={factor_df[sc].std():.4f}")

    print("\n" + "=" * 60)
    print("因子计算模块测试完成!")
    print("=" * 60)
