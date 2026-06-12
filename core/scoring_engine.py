#!/usr/bin/env python3
"""
Mini-GRP 评分排名模块 (Scoring Engine)

参考 Principal Global Investors 的 GRP 设计理念，
采用多因子、多维度加权评分框架进行量化排名。

评分框架 (GRP v3.2):
- 价值因子 (Value): 25% 权重
- 质量因子 (Quality): 25% 权重
- 增长因子 (Growth): 15% 权重
- 动量因子 (Momentum): 15% 权重
- 预期差距 (Expectation): 20% 权重 ← 新增，GRP 核心条件之一

评分方法:
1. 按维度内等权计算维度得分
2. 按维度权重计算综合得分（动态归一化，兼容缺失维度）
3. 转换为 0-100 百分位排名
4. 行业内排名
"""

import logging

import pandas as pd
import numpy as np
from typing import Dict, List

logger = logging.getLogger(__name__)

# =============================================================================
# GRP 风格的权重配置
# =============================================================================

# 基础权重配置（5维度）
# 当所有维度都存在时使用这些权重
# 当某个维度缺失时，composite_score 会自动重新归一化
FACTOR_WEIGHTS = {
    'value': 0.25,        # 价值因子权重（从30%下调）
    'quality': 0.25,      # 质量因子权重（从30%下调）
    'growth': 0.15,       # 增长因子权重（从20%下调）
    'momentum': 0.15,     # 动量因子权重（从20%下调）
    'expectation': 0.20,   # 预期差距因子权重（新增）
}

# 因子到维度的映射（使用 _z 后缀匹配 factor_engine 输出）
# 每个维度内的因子在计算维度得分时采用等权平均
FACTOR_TO_DIMENSION = {
    # === 价值因子 (25%) ===
    'pe_ttm_z': 'value',
    'pb_lf_z': 'value',
    'ps_ttm_z': 'value',
    'ev_ebitda_z': 'value',
    'dividend_yield_z': 'value',
    # === 质量因子 (25%) ===
    'roe_deducted_z': 'quality',
    'roa_z': 'quality',
    'gross_margin_z': 'quality',
    'net_margin_z': 'quality',
    'debt_to_equity_z': 'quality',
    # === 增长因子 (15%) ===
    'revenue_yoy_z': 'growth',
    'profit_yoy_z': 'growth',
    'fcf_yield_z': 'growth',
    # === 动量因子 (15%) ===
    'return_1m_z': 'momentum',
    'return_3m_z': 'momentum',
    'return_12m_z': 'momentum',
    # === 预期差距因子 (20%) ===
    'sue_z': 'expectation',
    'eps_revision_z': 'expectation',
    'rating_revision_z': 'expectation',
}

# 维度名称映射（中文）
DIMENSION_NAMES = {
    'value': '价值',
    'quality': '质量',
    'growth': '增长',
    'momentum': '动量',
    'expectation': '预期差距',
}

# 维度得分列名（动态检测，兼容4维度和5维度）
ALL_DIMENSION_SCORE_COLS = [
    'value_score', 'quality_score', 'growth_score',
    'momentum_score', 'expectation_score'
]


# =============================================================================
# 核心评分函数
# =============================================================================

def score_by_dimension(factor_df: pd.DataFrame) -> pd.DataFrame:
    """
    按维度计算得分

    对每个维度内的所有因子取等权平均，得到该维度的得分。
    维度得分经过标准化处理（z-score），均值为0，标准差为1。

    动态检测：只处理实际存在于输入数据中的因子列，
    缺失的维度记录 warning 但继续运行（得分设为 NaN）。

    Parameters
    ----------
    factor_df : pd.DataFrame
        包含因子列的DataFrame，需要有 FACTOR_TO_DIMENSION 中定义的因子列

    Returns
    -------
    pd.DataFrame
        原始数据 + 各维度得分列（如 value_score, quality_score, ...）
    """
    df = factor_df.copy()

    # 检查可用的因子列（动态检测，不要求全部存在）
    available_factors = set(df.columns) & set(FACTOR_TO_DIMENSION.keys())
    missing_factors = set(FACTOR_TO_DIMENSION.keys()) - available_factors

    if missing_factors:
        logger.warning("缺少以下因子列（将跳过）: %s", sorted(missing_factors))

    if not available_factors:
        raise ValueError(
            f"没有可用的因子列. 可用列: {sorted(df.columns)}"
        )

    # 按维度分组计算等权平均
    dimension_factors: Dict[str, List[str]] = {}
    for factor, dimension in FACTOR_TO_DIMENSION.items():
        if factor in available_factors:
            dimension_factors.setdefault(dimension, []).append(factor)

    # 计算每个维度的得分
    for dimension, factors in dimension_factors.items():
        score_col = f"{dimension}_score"

        # 维度内等权平均
        dimension_values = df[factors].mean(axis=1)

        # 标准化 (z-score)
        if 'market' in df.columns and df['market'].nunique() > 1:
            df[score_col] = dimension_values.groupby(df['market']).transform(
                lambda values: (values - values.mean()) / values.std()
                if values.std() > 0 else 0.0
            )
        else:
            mean = dimension_values.mean()
            std = dimension_values.std()
            if std > 0:
                df[score_col] = (dimension_values - mean) / std
            else:
                df[score_col] = 0.0

        logger.debug("维度 %s 得分已计算 (%d 个因子)", dimension, len(factors))

    # 对于没有可用因子的维度，设为 NaN（后续 composite_score 会处理）
    for dim in FACTOR_WEIGHTS.keys():
        score_col = f"{dim}_score"
        if score_col not in df.columns:
            df[score_col] = np.nan
            logger.info("维度 %s 无可用因子，得分设为 NaN", dim)

    return df


def _get_active_dimensions(scored_df: pd.DataFrame) -> List[str]:
    """
    获取实际存在的维度列表（非 NaN 列）。

    Returns
    -------
    List[str]
        实际存在的维度名称列表
    """
    active = []
    for dim in FACTOR_WEIGHTS.keys():
        col = f"{dim}_score"
        if col in scored_df.columns and not scored_df[col].isna().all():
            active.append(dim)
    return active


def _normalize_weights(active_dims: List[str]) -> Dict[str, float]:
    """
    对活跃维度的权重进行归一化。

    当某些维度缺失时，重新分配权重使得总和为1。
    例如：缺少 expectation 时，value=30/85, quality=30/85, growth=20/85, momentum=20/85

    Parameters
    ----------
    active_dims : List[str]
        实际存在的维度列表

    Returns
    -------
    Dict[str, float]
        归一化后的权重字典
    """
    total_weight = sum(FACTOR_WEIGHTS[d] for d in active_dims)
    if total_weight == 0:
        # 所有维度都缺失，平均分配
        n = len(active_dims) if active_dims else 1
        return {d: 1.0 / n for d in active_dims}

    return {d: FACTOR_WEIGHTS[d] / total_weight for d in active_dims}


def composite_score(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算综合评分

    按 GRP 权重配置计算加权综合得分。
    支持动态维度：当某些维度缺失时，自动重新归一化权重。

    然后将综合得分转换为 0-100 的百分位排名，便于直观比较。

    Parameters
    ----------
    scored_df : pd.DataFrame
        包含维度得分列的DataFrame (value_score, quality_score, etc.)

    Returns
    -------
    pd.DataFrame
        原始数据 + composite_score (0-100之间的百分位排名值)
    """
    df = scored_df.copy()

    # 检测实际存在的维度
    active_dims = _get_active_dimensions(df)

    if not active_dims:
        raise ValueError("没有可用的维度得分列。请先调用 score_by_dimension() 计算维度得分。")

    # 归一化权重
    normalized_weights = _normalize_weights(active_dims)

    logger.debug("活跃维度: %s", active_dims)
    logger.debug("归一化权重: %s", normalized_weights)

    # 计算加权综合得分（原始分数）
    # 缺失的维度得分用 0 填充（即不参与贡献）
    composite_raw = pd.Series(0.0, index=df.index)
    for dim in active_dims:
        score_col = f"{dim}_score"
        weight = normalized_weights[dim]
        # 用 0 填充 NaN（缺失维度不贡献也不惩罚）
        scores = df[score_col].fillna(0.0)
        composite_raw += scores * weight

    df['composite_score_raw'] = composite_raw

    # 转换为 0-100 的百分位排名
    df['composite_score'] = (
        df['composite_score_raw']
        .rank(method='average', pct=True, ascending=True) * 100
    )

    # 四舍五入到2位小数
    df['composite_score'] = df['composite_score'].round(2)

    return df


def rank_within_industry(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    行业内排名

    在每个行业（sw_industry_name）内部按 composite_score 进行排名，
    模拟 GRP 的行业特定模型。排名1表示该行业内得分最高。

    Parameters
    ----------
    scored_df : pd.DataFrame
        包含 composite_score 和 sw_industry_name 列的DataFrame

    Returns
    -------
    pd.DataFrame
        原始数据 + industry_rank (行业内的排名，1为最高)
    """
    df = scored_df.copy()

    # 检查必要列
    if 'composite_score' not in df.columns:
        raise ValueError(
            "缺少 composite_score 列. 请先调用 composite_score() 计算综合得分。"
        )
    if 'sw_industry_name' not in df.columns:
        raise ValueError(
            "缺少 sw_industry_name 列. 需要行业分类数据进行行业内排名。"
        )

    # 按行业分组，在每个行业内按 composite_score 降序排名
    rank_groups = ['sw_industry_name']
    if 'market' in df.columns and df['market'].nunique() > 1:
        rank_groups.insert(0, 'market')
    industry_ranks = df.groupby(rank_groups)['composite_score'].rank(
        method='min', ascending=False
    )
    # 处理可能的 NaN（某些行业无数据时）
    industry_ranks = industry_ranks.fillna(999)
    df['industry_rank'] = industry_ranks.astype(int)

    return df


def get_top_picks(scored_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    获取 Top N 推荐

    按 composite_score 排序取前 N 名，返回关键信息列。

    Parameters
    ----------
    scored_df : pd.DataFrame
        包含完整评分数据的DataFrame
    n : int, optional
        返回的股票数量，默认20

    Returns
    -------
    pd.DataFrame
        包含以下列的DataFrame:
        - code: 股票代码
        - name: 股票名称
        - sw_industry_name: 申万行业名称
        - composite_score: 综合得分 (0-100)
        - value_score: 价值维度得分
        - quality_score: 质量维度得分
        - growth_score: 增长维度得分
        - momentum_score: 动量维度得分
        - expectation_score: 预期差距维度得分（如果存在）
        - industry_rank: 行业内排名
    """
    # 基础必要列
    required_cols = ['code', 'name', 'sw_industry_name', 'composite_score']

    # 动态检测存在的维度得分列
    dim_cols = [c for c in ALL_DIMENSION_SCORE_COLS if c in scored_df.columns]

    missing_base = [c for c in required_cols if c not in scored_df.columns]
    if missing_base:
        raise ValueError(f"缺少必要列: {missing_base}")

    # 选择需要的列
    output_cols = required_cols + dim_cols
    if 'industry_rank' in scored_df.columns:
        output_cols.append('industry_rank')

    # 按 composite_score 降序排序，取前N名
    top = (scored_df[output_cols]
           .sort_values('composite_score', ascending=False)
           .head(n)
           .reset_index(drop=True))

    return top


# =============================================================================
# 辅助函数
# =============================================================================

def get_dimension_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    获取各维度的统计摘要

    Parameters
    ----------
    scored_df : pd.DataFrame
        包含维度得分的DataFrame

    Returns
    -------
    pd.DataFrame
        各维度的均值、标准差、最小值、最大值等统计信息
    """
    score_cols = [c for c in ALL_DIMENSION_SCORE_COLS if c in scored_df.columns]
    if not score_cols:
        return pd.DataFrame()

    summary = scored_df[score_cols].describe().T
    summary.index = [DIMENSION_NAMES.get(c.replace('_score', ''), c)
                     for c in summary.index]
    return summary


def get_industry_stats(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    获取各行业统计信息

    Parameters
    ----------
    scored_df : pd.DataFrame
        包含行业分类和维度得分的DataFrame

    Returns
    -------
    pd.DataFrame
        各行业在各维度上的平均得分
    """
    if 'sw_industry_name' not in scored_df.columns:
        return pd.DataFrame()

    score_cols = [c for c in ALL_DIMENSION_SCORE_COLS if c in scored_df.columns]
    if not score_cols:
        return pd.DataFrame()

    industry_stats = scored_df.groupby('sw_industry_name')[score_cols].mean()
    industry_stats.columns = [DIMENSION_NAMES.get(c.replace('_score', ''), c)
                              for c in industry_stats.columns]
    industry_stats['股票数量'] = scored_df.groupby('sw_industry_name').size()

    # 按综合得分排序
    if 'composite_score' in scored_df.columns:
        industry_stats['平均综合得分'] = (
            scored_df.groupby('sw_industry_name')['composite_score'].mean()
        )
        industry_stats = industry_stats.sort_values('平均综合得分', ascending=False)

    return industry_stats.round(4)


# =============================================================================
# 全量评分管道 (Pipeline)
# =============================================================================

def run_full_scoring(factor_df: pd.DataFrame) -> pd.DataFrame:
    """
    执行完整的评分管道

    依次执行: 维度评分 -> 综合评分 -> 行业内排名

    Parameters
    ----------
    factor_df : pd.DataFrame
        包含因子列的原始数据

    Returns
    -------
    pd.DataFrame
        完整的评分结果DataFrame
    """
    print("[scoring_engine] Step 1/3: 计算维度得分...")
    scored = score_by_dimension(factor_df)
    active_dims = _get_active_dimensions(scored)
    print(f"  - 维度得分已计算: {[f'{d}_score' for d in active_dims]}")

    print("[scoring_engine] Step 2/3: 计算综合评分...")
    scored = composite_score(scored)
    print(f"  - 综合得分范围: {scored['composite_score'].min():.2f} - {scored['composite_score'].max():.2f}")

    if 'sw_industry_name' in scored.columns:
        print("[scoring_engine] Step 3/3: 行业内排名...")
        scored = rank_within_industry(scored)
        print(f"  - 已按行业排名")
    else:
        print("[scoring_engine] Step 3/3: 跳过行业内排名 (缺少 sw_industry_name)")

    print("[scoring_engine] 评分完成!")
    return scored


if __name__ == '__main__':
    # 测试评分模块
    from mock_data import generate_mock_data
    import numpy as np

    print("=" * 60)
    print("测试评分模块 (5维度 + 4维度兼容)")
    print("=" * 60)

    # ===== 测试 1: 5维度完整数据 =====
    print("\n[测试 1] 5维度完整数据 (含预期差距因子)")
    df = generate_mock_data(n_stocks=100)

    # 添加模拟的预期差距因子
    np.random.seed(42)
    df['sue'] = np.random.normal(0, 1, len(df))
    df['eps_revision'] = np.random.normal(0, 0.5, len(df))
    df['rating_revision'] = np.random.normal(0, 0.3, len(df))

    # 添加标准化列（模拟 factor_engine 输出）
    from factor_engine import standardize
    df['sue_z'] = standardize(df['sue'])
    df['eps_revision_z'] = standardize(df['eps_revision'])
    df['rating_revision_z'] = standardize(df['rating_revision'])

    scored = run_full_scoring(df)
    top20 = get_top_picks(scored, n=20)
    print(f"\nTop 5 (5维度):")
    print(top20[['code', 'name', 'sw_industry_name', 'composite_score',
                 'value_score', 'quality_score', 'growth_score',
                 'momentum_score', 'expectation_score']].head())

    # ===== 测试 2: 4维度向后兼容 =====
    print("\n[测试 2] 4维度数据 (不含预期差距因子，向后兼容)")
    df4 = generate_mock_data(n_stocks=100)
    scored4 = run_full_scoring(df4)
    top20_4 = get_top_picks(scored4, n=20)
    print(f"\nTop 5 (4维度):")
    print(top20_4[['code', 'name', 'sw_industry_name', 'composite_score',
                   'value_score', 'quality_score', 'growth_score',
                   'momentum_score']].head())

    # 验证权重归一化
    active4 = _get_active_dimensions(scored4)
    weights4 = _normalize_weights(active4)
    print(f"\n4维度归一化权重: {weights4}")
    expected_sum = sum(weights4.values())
    print(f"权重总和: {expected_sum:.4f} (应为 1.0)")
    assert abs(expected_sum - 1.0) < 1e-6, "权重归一化错误!"

    # ===== 测试 3: 维度摘要 =====
    print("\n[测试 3] 维度得分统计 (5维度):")
    print(get_dimension_summary(scored))

    print("\n[测试 4] 维度得分统计 (4维度):")
    print(get_dimension_summary(scored4))

    # ===== 测试 5: 行业统计 =====
    print("\n[测试 5] 行业统计 (5维度):")
    print(get_industry_stats(scored).head(5))

    print("\n" + "=" * 60)
    print("评分模块测试完成! 5维度 + 4维度兼容均通过")
    print("=" * 60)
