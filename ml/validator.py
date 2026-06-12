"""
传统横截面因子检验模块
Factor Validation Pipeline — v3.1核心升级

所有因子在进入ML模块之前，必须通过以下检验：
1. Rank IC (Spearman Rank Correlation) — 因子暴露与forward return排名的相关性
2. ICIR (IC Information Ratio) — IC的稳定性 = mean(IC)/std(IC)
3. Quantile Spread — Q5(top) - Q1(bottom)的收益差
4. Monotonicity Test — 五分位收益是否单调递增(Kendall's tau)
5. Factor Decay — IC半衰期分析

依赖降级策略:
- 无外部依赖，纯numpy/pandas实现
- Spearman秩相关用numpy实现（不依赖scipy）
- Kendall's tau用简化实现
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

__all__ = [
    "calculate_rank_ic",
    "calculate_icir",
    "calculate_quantile_spread",
    "calculate_monotonicity",
    "calculate_factor_decay",
    "FactorValidationReport",
]


# ---------------------------------------------------------------------------
# 低层工具函数
# ---------------------------------------------------------------------------

def _rank_data(a: np.ndarray) -> np.ndarray:
    """
    计算1-D数组的秩（排名），平均法处理并列。

    Parameters
    ----------
    a : np.ndarray
        输入数组，可能含NaN。

    Returns
    -------
    np.ndarray
        秩数组，NaN位置保持NaN。
    """
    a = np.asarray(a, dtype=float)
    mask = ~np.isnan(a)
    ranked = np.empty_like(a, dtype=float)
    ranked[:] = np.nan
    if mask.sum() == 0:
        return ranked

    valid = a[mask]
    sorter = np.argsort(valid, kind="mergesort")
    inv = np.empty_like(sorter, dtype=float)
    inv[sorter] = np.arange(len(valid), dtype=float)

    # 平均法处理并列
    unique, counts = np.unique(valid, return_counts=True)
    for val, cnt in zip(unique, counts):
        if cnt > 1:
            idx = np.where(valid == val)[0]
            avg_rank = inv[idx].mean()
            inv[idx] = avg_rank

    ranked[mask] = inv + 1.0  # 1-based rank
    return ranked


def _spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    纯numpy实现的Spearman秩相关系数。

    Parameters
    ----------
    x, y : np.ndarray
        两个一维数组，允许含NaN。

    Returns
    -------
    float
        Spearman秩相关系数，若无法计算返回np.nan。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (~np.isnan(x)) & (~np.isnan(y))
    n = mask.sum()
    if n < 3:
        return np.nan

    rx = _rank_data(x[mask])
    ry = _rank_data(y[mask])

    d = rx - ry
    # 使用Pearson公式计算秩相关系数（更稳健）
    rx_mean, ry_mean = rx.mean(), ry.mean()
    num = ((rx - rx_mean) * (ry - ry_mean)).sum()
    den = np.sqrt(((rx - rx_mean) ** 2).sum() * ((ry - ry_mean) ** 2).sum())
    if den == 0:
        return 0.0
    return float(num / den)


# ---------------------------------------------------------------------------
# 核心检验函数
# ---------------------------------------------------------------------------

def calculate_rank_ic(factor_values: pd.Series, forward_returns: pd.Series) -> float:
    """
    计算Spearman秩相关系数（因子值 vs 未来收益排名）。

    Rank IC衡量因子暴露与未来收益排名的单调关系，取值[-1, 1]。
    |Rank IC| > 0.03 通常被认为有预测能力。

    Parameters
    ----------
    factor_values : pd.Series
        单期因子值序列，index为股票代码。
    forward_returns : pd.Series
        对应的前向收益序列，index为股票代码。

    Returns
    -------
    float
        Spearman秩相关系数（Rank IC），含NaN时自动剔除。

    Examples
    --------
    >>> ic = calculate_rank_ic(factor_df['pe_ratio'], factor_df['return_1m_fwd'])
    >>> print(f"Rank IC = {ic:.4f}")
    """
    aligned_factor, aligned_returns = factor_values.align(forward_returns, join="inner")
    return _spearman_rank_corr(aligned_factor.values, aligned_returns.values)


def calculate_icir(ic_series: pd.Series) -> float:
    """
    计算IC Information Ratio = mean(IC) / std(IC)。

    ICIR衡量因子IC的稳定性。ICIR > 0.5 表示因子具有统计显著性，
    ICIR > 1.0 表示因子非常稳定可靠。

    Parameters
    ----------
    ic_series : pd.Series
        多期IC时间序列（如每月一个Rank IC值）。

    Returns
    -------
    float
        ICIR值。若标准差为0返回np.nan。

    Examples
    --------
    >>> monthly_ic = factor_df.groupby('date').apply(
    ...     lambda g: calculate_rank_ic(g['pe'], g['fwd_return'])
    ... )
    >>> icir = calculate_icir(monthly_ic)
    """
    clean = ic_series.dropna()
    if len(clean) < 2:
        return np.nan
    mean_ic = clean.mean()
    std_ic = clean.std(ddof=1)
    if std_ic == 0:
        return np.nan
    return float(mean_ic / std_ic)


def calculate_quantile_spread(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    n_quantiles: int = 5,
) -> float:
    """
    将股票按因子值分为n组，返回top组平均收益 - bottom组平均收益。

    Quantile Spread > 0 表示因子具有区分好坏股票的能力。
    对于正向因子，Q5 - Q1 > 0 表示因子值越大未来收益越好。

    Parameters
    ----------
    factor_values : pd.Series
        单期因子值序列。
    forward_returns : pd.Series
        对应前向收益序列。
    n_quantiles : int, default 5
        分位数分组数量。

    Returns
    -------
    float
        Top组平均收益 - Bottom组平均收益。

    Examples
    --------
    >>> spread = calculate_quantile_spread(factors['pe'], returns['1m_fwd'], n_quantiles=5)
    >>> print(f"Q5-Q1 Spread = {spread:.4f}")
    """
    aligned_factor, aligned_returns = factor_values.align(forward_returns, join="inner")
    aligned_factor = aligned_factor.dropna()
    aligned_returns = aligned_returns.loc[aligned_factor.index]

    if len(aligned_factor) < n_quantiles:
        return np.nan

    # 使用pandas qcut进行分位
    try:
        labels = list(range(1, n_quantiles + 1))
        quantiles = pd.qcut(aligned_factor, q=n_quantiles, labels=labels, duplicates="drop")
    except ValueError:
        # 因子值过于集中时fallback
        return np.nan

    grouped = aligned_returns.groupby(quantiles, observed=True).mean()
    if len(grouped) < n_quantiles:
        return np.nan

    top_return = grouped.iloc[-1]  # 最高quantile
    bottom_return = grouped.iloc[0]  # 最低quantile
    return float(top_return - bottom_return)


def calculate_monotonicity(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    n_quantiles: int = 5,
) -> float:
    """
    计算五分位收益的单调性系数（简化版Kendall's tau）。

    检查相邻quantile的收益是否单调递增。tau = 1 表示完全单调递增，
    tau = -1 表示完全单调递减，tau = 0 表示无单调关系。

    Parameters
    ----------
    factor_values : pd.Series
        单期因子值序列。
    forward_returns : pd.Series
        对应前向收益序列。
    n_quantiles : int, default 5
        分位数分组数量。

    Returns
    -------
    float
        简化Kendall's tau单调性系数。
    """
    aligned_factor, aligned_returns = factor_values.align(forward_returns, join="inner")
    aligned_factor = aligned_factor.dropna()
    aligned_returns = aligned_returns.loc[aligned_factor.index]

    if len(aligned_factor) < n_quantiles:
        return np.nan

    try:
        labels = list(range(1, n_quantiles + 1))
        quantiles = pd.qcut(aligned_factor, q=n_quantiles, labels=labels, duplicates="drop")
    except ValueError:
        return np.nan

    grouped = aligned_returns.groupby(quantiles, observed=True).mean()
    if len(grouped) < 2:
        return np.nan

    quantile_returns = grouped.values

    # 计算简化版Kendall's tau：统计单调对的数量
    concordant = 0
    discordant = 0
    n_pairs = 0
    for i in range(len(quantile_returns)):
        for j in range(i + 1, len(quantile_returns)):
            if quantile_returns[j] > quantile_returns[i]:
                concordant += 1
            elif quantile_returns[j] < quantile_returns[i]:
                discordant += 1
            n_pairs += 1

    if n_pairs == 0:
        return 0.0
    return float((concordant - discordant) / n_pairs)


def calculate_factor_decay(
    factor_values_history: pd.DataFrame,
    forward_returns: pd.Series,
    max_lag: int = 12,
) -> pd.Series:
    """
    因子衰减分析：计算不同lag下的Rank IC。

    因子衰减速度反映因子的半衰期。若lag=1的IC=0.05，lag=6的IC=0.025，
    则半衰期约为6个月。

    Parameters
    ----------
    factor_values_history : pd.DataFrame
        历史因子值矩阵，index为日期，columns为股票代码。
    forward_returns : pd.Series
        前向收益序列，MultiIndex为(日期, 股票代码)。
    max_lag : int, default 12
        最大滞后月数。

    Returns
    -------
    pd.Series
        lag=1到max_lag的Rank IC序列，index为lag值。

    Examples
    --------
    >>> decay = calculate_factor_decay(factor_history, forward_returns, max_lag=12)
    >>> half_life = decay[decay >= decay.iloc[0] * 0.5].index[-1]
    >>> print(f"IC半衰期 = {half_life}个月")
    """
    ic_by_lag = {}
    factor_dates = factor_values_history.index

    for lag in range(1, max_lag + 1):
        ic_list = []
        for date in factor_values_history.index:
            # 找到lag期后的日期
            future_idx = factor_dates.get_loc(date) + lag
            if future_idx >= len(factor_dates):
                continue
            future_date = factor_dates[future_idx]

            factor_vals = factor_values_history.loc[date].dropna()
            try:
                fwd_ret = forward_returns.loc[future_date].dropna()
            except KeyError:
                continue

            aligned_factor, aligned_ret = factor_vals.align(fwd_ret, join="inner")
            if len(aligned_factor) < 3:
                continue
            ic = _spearman_rank_corr(aligned_factor.values, aligned_ret.values)
            if not np.isnan(ic):
                ic_list.append(ic)

        if ic_list:
            ic_by_lag[lag] = float(np.mean(ic_list))
        else:
            ic_by_lag[lag] = np.nan

    return pd.Series(ic_by_lag, name="rank_ic")


# ---------------------------------------------------------------------------
# 报告生成类
# ---------------------------------------------------------------------------

class FactorValidationReport:
    """
    生成完整的因子检验报告。

    对每个因子运行完整四项检验（Rank IC、ICIR、Quantile Spread、Monotonicity），
    并根据预设阈值判断因子是否通过检验。

    Parameters
    ----------
    None

    Attributes
    ----------
    THRESHOLDS : dict
        各项检验的通过阈值。
    results : pd.DataFrame
        检验结果表格。
    """

    THRESHOLDS = {
        "min_rank_ic": 0.03,  # |Rank IC|最低有效门槛
        "min_icir": 0.5,  # ICIR稳定性门槛
        "min_quantile_spread": 0,  # Quantile Spread必须为正
        "min_monotonicity": 0.3,  # 单调性最低门槛
    }

    def __init__(self) -> None:
        self.results: Optional[pd.DataFrame] = None

    def run_full_validation(
        self,
        factor_df: pd.DataFrame,
        factor_cols: List[str],
        return_col: str = "forward_return",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """
        对每个因子运行完整四项检验。

        Parameters
        ----------
        factor_df : pd.DataFrame
            包含因子值和前向收益的DataFrame。
        factor_cols : List[str]
            待检验的因子列名列表。
        return_col : str, default 'forward_return'
            前向收益列名。
        date_col : str, default 'date'
            日期列名。

        Returns
        -------
        pd.DataFrame
            检验结果表格，列为:
            factor | rank_ic | icir | quantile_spread | monotonicity | pass | reason
        """
        if date_col not in factor_df.columns:
            raise ValueError(f"date_col '{date_col}' not found in factor_df")
        if return_col not in factor_df.columns:
            raise ValueError(f"return_col '{return_col}' not found in factor_df")

        records = []

        for factor_col in factor_cols:
            if factor_col not in factor_df.columns:
                records.append(
                    {
                        "factor": factor_col,
                        "rank_ic": np.nan,
                        "icir": np.nan,
                        "quantile_spread": np.nan,
                        "monotonicity": np.nan,
                        "pass": False,
                        "reason": "Column not found in data",
                    }
                )
                continue

            # 1. 计算每期Rank IC
            ic_list = []
            for date, group in factor_df.groupby(date_col):
                factor_vals = group[factor_col]
                fwd_ret = group[return_col]
                ic = calculate_rank_ic(factor_vals, fwd_ret)
                if not np.isnan(ic):
                    ic_list.append(ic)

            if len(ic_list) < 2:
                records.append(
                    {
                        "factor": factor_col,
                        "rank_ic": np.nan,
                        "icir": np.nan,
                        "quantile_spread": np.nan,
                        "monotonicity": np.nan,
                        "pass": False,
                        "reason": "Insufficient data for IC calculation",
                    }
                )
                continue

            ic_series = pd.Series(ic_list)
            mean_rank_ic = ic_series.mean()
            icir = calculate_icir(ic_series)

            # 2. 计算Quantile Spread（取所有日期的平均）
            spreads = []
            mono_list = []
            for date, group in factor_df.groupby(date_col):
                spread = calculate_quantile_spread(
                    group[factor_col], group[return_col]
                )
                mono = calculate_monotonicity(group[factor_col], group[return_col])
                if not np.isnan(spread):
                    spreads.append(spread)
                if not np.isnan(mono):
                    mono_list.append(mono)

            mean_spread = np.mean(spreads) if spreads else np.nan
            mean_mono = np.mean(mono_list) if mono_list else np.nan

            # 3. 判断是否通过
            pass_flag = True
            reasons = []

            if abs(mean_rank_ic) < self.THRESHOLDS["min_rank_ic"]:
                pass_flag = False
                reasons.append(
                    f"|Rank IC|={abs(mean_rank_ic):.4f} < {self.THRESHOLDS['min_rank_ic']}"
                )

            if abs(icir) < self.THRESHOLDS["min_icir"]:
                pass_flag = False
                reasons.append(
                    f"|ICIR|={abs(icir):.4f} < {self.THRESHOLDS['min_icir']}"
                )

            if mean_spread < self.THRESHOLDS["min_quantile_spread"]:
                pass_flag = False
                reasons.append(
                    f"Spread={mean_spread:.4f} < {self.THRESHOLDS['min_quantile_spread']}"
                )

            if abs(mean_mono) < self.THRESHOLDS["min_monotonicity"]:
                pass_flag = False
                reasons.append(
                    f"|Monotonicity|={abs(mean_mono):.4f} < {self.THRESHOLDS['min_monotonicity']}"
                )

            reason_str = "; ".join(reasons) if reasons else "Passed all tests"

            records.append(
                {
                    "factor": factor_col,
                    "rank_ic": round(mean_rank_ic, 4),
                    "icir": round(icir, 4),
                    "quantile_spread": round(mean_spread, 4),
                    "monotonicity": round(mean_mono, 4),
                    "pass": pass_flag,
                    "reason": reason_str,
                }
            )

        self.results = pd.DataFrame(records)
        return self.results

    def get_passed_factors(self) -> List[str]:
        """
        返回通过全部检验的因子列表。

        Returns
        -------
        List[str]
            通过检验的因子名称列表。

        Raises
        ------
        RuntimeError
            若尚未运行run_full_validation。
        """
        if self.results is None:
            raise RuntimeError("Must run run_full_validation() first")
        return self.results[self.results["pass"]]["factor"].tolist()

    def get_failed_factors(self) -> List[Tuple[str, str]]:
        """
        返回未通过的因子及原因。

        Returns
        -------
        List[Tuple[str, str]]
            [(factor_name, reason), ...]

        Raises
        ------
        RuntimeError
            若尚未运行run_full_validation。
        """
        if self.results is None:
            raise RuntimeError("Must run run_full_validation() first")
        failed = self.results[~self.results["pass"]]
        return list(zip(failed["factor"], failed["reason"]))

    def summary(self) -> Dict:
        """
        返回检验摘要统计。

        Returns
        -------
        Dict
            {total: int, passed: int, failed: int, pass_rate: float,
             avg_rank_ic: float, avg_icir: float}

        Raises
        ------
        RuntimeError
            若尚未运行run_full_validation。
        """
        if self.results is None:
            raise RuntimeError("Must run run_full_validation() first")

        total = len(self.results)
        passed = int(self.results["pass"].sum())
        failed = total - passed
        pass_rate = passed / total if total > 0 else 0.0

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(pass_rate, 4),
            "avg_rank_ic": round(float(self.results["rank_ic"].abs().mean()), 4),
            "avg_icir": round(float(self.results["icir"].abs().mean()), 4),
        }


# ---------------------------------------------------------------------------
# 用法示例
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    n_dates = 24
    n_stocks = 100
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="MS")
    stocks = [f"STK_{i:03d}" for i in range(n_stocks)]

    data_list = []
    for d in dates:
        for s in stocks:
            data_list.append(
                {
                    "date": d,
                    "stock": s,
                    "pe_ratio": np.random.randn(),
                    "pb_ratio": np.random.randn(),
                    "momentum_1m": np.random.randn(),
                    "volatility_20d": np.random.randn(),
                    "forward_return": np.random.randn() * 0.05,
                }
            )

    df = pd.DataFrame(data_list)
    factor_cols = ["pe_ratio", "pb_ratio", "momentum_1m", "volatility_20d"]

    # 运行因子检验
    report = FactorValidationReport()
    results = report.run_full_validation(df, factor_cols)
    print("=" * 60)
    print("Factor Validation Report v3.1")
    print("=" * 60)
    print(results.to_string(index=False))
    print()
    print("Summary:", report.summary())
    print("Passed Factors:", report.get_passed_factors())
