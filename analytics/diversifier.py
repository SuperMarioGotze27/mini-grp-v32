"""低相关性组合分散化 — Portfolio Diversification via Correlation
对应课程：Assignment 1 Portfolio Management (L2-17)

核心思想：
    传统的top-down选股只考虑单只股票的质量（如composite score），
    忽略了组合层面的相关性结构。本模块通过贪心算法，在选股时
    优先选择与已选股票平均相关性最低的新股票，从而降低组合
    整体风险，提升diversification ratio。

课程对应：
    - L2-17 Portfolio Construction: 分散化是降低非系统性风险的核心手段
    - Assignment 1: 从单因子排序扩展到组合层面的相关性优化
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd

__all__ = [
    "diversify_by_correlation",
    "diversify_with_score_penalty",
    "get_portfolio_turnover",
    "correlation_matrix_heatmap",
    "CorrelationDiversifier",
]


def diversify_by_correlation(
    scored_stocks: pd.DataFrame,
    returns: pd.DataFrame,
    n_select: int = 10,
    score_col: str = "composite",
) -> List[str]:
    """
    贪心算法：每次选与已选股票平均绝对相关性最低的下一支

    算法步骤：
        1. 按 composite score 降序排序，选第一名作为起点
        2. 对每一只候选股票，计算其与已选股票的平均绝对相关性
        3. 选择平均绝对相关性最低的股票加入组合
        4. 重复步骤2-3直到选够 n_select 只

    Parameters
    ----------
    scored_stocks : pd.DataFrame
        含 composite 得分的股票列表，index 为 ticker。
    returns : pd.DataFrame
        历史收益率矩阵，columns 为 ticker。
    n_select : int, default 10
        目标选股数量。
    score_col : str, default 'composite'
        得分列名，用于第一步选择得分最高的股票作为起点。

    Returns
    -------
    List[str]
        选出的股票 ticker 列表（按选择顺序）。

    Notes
    -----
    - 第一步用 score 选起点，后续步骤忽略 score，只看相关性
    - 如果 returns 中缺少某只股票的收益数据，则跳过该股票
    - 时间复杂度：O(n_select * n_candidates * n_periods)

    Examples
    --------
    >>> selected = diversify_by_correlation(scored_df, returns_df, n_select=5)
    >>> print(selected)
    ['AAPL', 'JNJ', 'XOM', 'PG', 'KO']
    """
    # 获取有效的候选股票列表（要求得分和收益数据都可用）
    candidates = scored_stocks[score_col].dropna().sort_values(ascending=False)
    valid_tickers = [t for t in candidates.index if t in returns.columns]

    if len(valid_tickers) == 0:
        return []

    # 第一步：选得分最高的股票作为起点
    selected: List[str] = [valid_tickers[0]]
    remaining = [t for t in valid_tickers if t not in selected]

    corr_matrix = returns.corr()

    # 贪心循环：每次选平均绝对相关性最低的
    while len(selected) < n_select and remaining:
        best_ticker = None
        best_avg_corr = float("inf")

        for ticker in remaining:
            if ticker not in corr_matrix.columns:
                continue
            # 计算与所有已选股票的平均绝对相关性
            avg_corr = corr_matrix.loc[ticker, selected].abs().mean()
            if avg_corr < best_avg_corr:
                best_avg_corr = avg_corr
                best_ticker = ticker

        if best_ticker is None:
            break

        selected.append(best_ticker)
        remaining.remove(best_ticker)

    return selected


def diversify_with_score_penalty(
    scored_stocks: pd.DataFrame,
    returns: pd.DataFrame,
    n_select: int = 10,
    lambda_corr: float = 0.3,
) -> pd.DataFrame:
    """
    双目标优化：max score - lambda * avg_correlation

    对每只股票计算 adjusted_score = score - lambda * avg_corr_with_candidates,
    然后按 adjusted_score 降序选出 n_select 只。

    Parameters
    ----------
    scored_stocks : pd.DataFrame
        含 composite 得分的股票列表，index 为 ticker。
    returns : pd.DataFrame
        历史收益率矩阵，columns 为 ticker。
    n_select : int, default 10
        目标选股数量。
    lambda_corr : float, default 0.3
        相关性惩罚系数。越大表示越重视分散化（低相关性），
        越小表示越重视单个股票的质量（高 score）。

    Returns
    -------
    pd.DataFrame
        columns: ticker | score | avg_corr | adjusted_score | selected
        按 adjusted_score 降序排列。

    Notes
    -----
    - 与 greedy 方法不同，这里一次性对所有候选股票做评估
    - avg_corr 是该股票与所有其他候选股票的平均绝对相关性
    - lambda_corr = 0 退化为纯 score 排序；lambda_corr -> inf 退化为纯低相关排序

    Examples
    --------
    >>> result = diversify_with_score_penalty(scored_df, returns_df, n_select=5, lambda_corr=0.5)
    >>> print(result.head())
              score   avg_corr  adjusted_score  selected
    AAPL      0.85       0.42          0.6400      True
    MSFT      0.82       0.45          0.5950      True
    """
    score_col = "composite" if "composite" in scored_stocks.columns else scored_stocks.columns[0]
    candidates = scored_stocks[score_col].dropna()
    valid_tickers = [t for t in candidates.index if t in returns.columns]

    if len(valid_tickers) == 0:
        return pd.DataFrame(
            columns=["score", "avg_corr", "adjusted_score", "selected"]
        )

    corr_matrix = returns[valid_tickers].corr().abs()

    rows = []
    for ticker in valid_tickers:
        score = candidates[ticker]
        # 与该股票与其他所有候选股票的平均绝对相关性
        other_tickers = [t for t in valid_tickers if t != ticker]
        if other_tickers:
            avg_corr = corr_matrix.loc[ticker, other_tickers].mean()
        else:
            avg_corr = 0.0
        adjusted_score = score - lambda_corr * avg_corr
        rows.append(
            {
                "ticker": ticker,
                "score": round(score, 4),
                "avg_corr": round(avg_corr, 4),
                "adjusted_score": round(adjusted_score, 4),
            }
        )

    result_df = pd.DataFrame(rows).sort_values("adjusted_score", ascending=False)
    result_df["selected"] = False
    n = min(n_select, len(result_df))
    result_df.iloc[:n, result_df.columns.get_loc("selected")] = True

    return result_df.reset_index(drop=True)


def get_portfolio_turnover(
    current_weights: Dict[str, float],
    new_weights: Dict[str, float],
) -> float:
    """
    计算调仓换手率 = sum(|new_weight_i - current_weight_i|) / 2

    Parameters
    ----------
    current_weights : Dict[str, float]
        当前持仓权重，key 为 ticker，value 为权重。
    new_weights : Dict[str, float]
        目标持仓权重，key 为 ticker，value 为权重。

    Returns
    -------
    float
        换手率，范围 [0, 1]。
        0 表示完全不变，1 表示全部卖出再买入（理论上界）。

    Examples
    --------
    >>> turnover = get_portfolio_turnover(
    ...     {"AAPL": 0.5, "MSFT": 0.5},
    ...     {"AAPL": 0.3, "GOOG": 0.7}
    ... )
    >>> print(turnover)
    0.4
    """
    all_tickers = set(current_weights.keys()) | set(new_weights.keys())

    turnover = 0.0
    for ticker in all_tickers:
        w_old = current_weights.get(ticker, 0.0)
        w_new = new_weights.get(ticker, 0.0)
        turnover += abs(w_new - w_old)

    return turnover / 2.0


def correlation_matrix_heatmap(
    returns: pd.DataFrame,
    selected: Optional[List[str]] = None,
) -> np.ndarray:
    """
    返回相关性矩阵（前端热力图用）

    Parameters
    ----------
    returns : pd.DataFrame
        历史收益率矩阵，columns 为 ticker。
    selected : Optional[List[str]], default None
        如果提供，只返回 selected 股票之间的相关性矩阵。

    Returns
    -------
    np.ndarray
        Pearson 相关性矩阵，形状 (n_selected, n_selected)。

    Examples
    --------
    >>> corr_mat = correlation_matrix_heatmap(returns_df, selected=['AAPL', 'MSFT', 'GOOG'])
    >>> print(corr_mat.shape)
    (3, 3)
    """
    if selected is not None:
        available = [t for t in selected if t in returns.columns]
        if len(available) == 0:
            return np.array([])
        subset = returns[available]
    else:
        subset = returns

    corr = subset.corr()
    return corr.values


class CorrelationDiversifier:
    """
    低相关性分散化选股 — 统一接口类

    将上述函数封装为类，方便在 Pipeline 中复用。

    Parameters
    ----------
    returns : pd.DataFrame
        历史收益率矩阵，columns 为 ticker，index 为日期。

    Attributes
    ----------
    returns : pd.DataFrame
        存储的收益率矩阵。
    corr_matrix : pd.DataFrame
        预计算的 Pearson 相关性矩阵。

    Examples
    --------
    >>> cd = CorrelationDiversifier(returns_df)
    >>> selected = cd.select_portfolio(candidates_df, n=5, method='greedy')
    >>> ratio = cd.get_diversification_ratio(selected)
    >>> print(f"Diversification Ratio: {ratio:.4f}")
    """

    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.corr_matrix = returns.corr()

    def select_portfolio(
        self,
        candidates: pd.DataFrame,
        n: int = 10,
        method: str = "greedy",
    ) -> List[str]:
        """
        选股主入口

        Parameters
        ----------
        candidates : pd.DataFrame
            候选股票 DataFrame，需包含 composite 得分列。
        n : int, default 10
            目标选股数量。
        method : str, default 'greedy'
            选股方法：
            - 'greedy': 贪心算法（ diversify_by_correlation ）
            - 'score_penalty': 双目标优化（ diversify_with_score_penalty ）

        Returns
        -------
        List[str]
            选出的 ticker 列表。
        """
        if method == "greedy":
            return diversify_by_correlation(candidates, self.returns, n_select=n)
        elif method == "score_penalty":
            result = diversify_with_score_penalty(
                candidates, self.returns, n_select=n
            )
            return result[result["selected"]]["ticker"].tolist()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'greedy' or 'score_penalty'.")

    def get_diversification_ratio(self, selected: List[str]) -> float:
        """
        计算分散化比率 Diversification Ratio

        Definition:
            DR = weighted_avg_volatility / portfolio_volatility

        DR > 1 表示组合有分散化收益（组合波动小于加权平均波动）。
        DR 越大，分散化效果越好。

        Parameters
        ----------
        selected : List[str]
            选中的 ticker 列表。

        Returns
        -------
        float
            分散化比率。如果选中的股票少于2只，返回 1.0。
        """
        available = [t for t in selected if t in self.returns.columns]
        if len(available) < 2:
            return 1.0

        subset = self.returns[available]
        vols = subset.std()
        corr = subset.corr()
        n = len(available)

        # 等权重
        w = np.ones(n) / n

        # 加权平均波动率
        weighted_avg_vol = float(w @ vols.values)

        # 组合波动率 = sqrt(w' * Sigma * w)
        cov = subset.cov()
        port_vol = float(np.sqrt(w @ cov.values @ w))

        if port_vol < 1e-12:
            return 1.0

        return weighted_avg_vol / port_vol


# ───────────────────────────────────────────────────────────────────────────────
# 用法示例（可独立运行）
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Demo: Correlation Diversifier")
    print("=" * 60)

    np.random.seed(42)

    # 构造模拟数据：10只股票，252个交易日
    n_stocks = 10
    n_days = 252
    tickers = [f"STK{i:02d}" for i in range(n_stocks)]

    # 生成相关性结构：market factor + idiosyncratic
    market = np.random.randn(n_days) * 0.01
    returns_data = {}
    for i, t in enumerate(tickers):
        beta = 0.3 + i * 0.05  # 不同 beta
        specific = np.random.randn(n_days) * 0.008
        returns_data[t] = beta * market + specific

    returns_df = pd.DataFrame(returns_data, index=pd.date_range("2023-01-01", periods=n_days))

    # 构造 composite score
    scores = np.random.uniform(0, 1, n_stocks)
    scored_df = pd.DataFrame({"composite": scores}, index=tickers)

    # 1) Greedy 选股
    selected_greedy = diversify_by_correlation(scored_df, returns_df, n_select=5)
    print(f"\n[1] Greedy selection (n=5): {selected_greedy}")

    # 2) Score Penalty 选股
    result_sp = diversify_with_score_penalty(scored_df, returns_df, n_select=5, lambda_corr=0.5)
    print(f"\n[2] Score Penalty selection:")
    print(result_sp.to_string(index=False))

    # 3) Turnover 计算
    current = {"STK00": 0.25, "STK01": 0.25, "STK02": 0.25, "STK03": 0.25}
    new = {"STK00": 0.4, "STK01": 0.3, "STK04": 0.3}
    turnover = get_portfolio_turnover(current, new)
    print(f"\n[3] Portfolio Turnover: {turnover:.4f}")

    # 4) Correlation Matrix
    corr_mat = correlation_matrix_heatmap(returns_df, selected=selected_greedy[:3])
    print(f"\n[4] Correlation Matrix (3x3):\n{np.round(corr_mat, 3)}")

    # 5) CorrelationDiversifier 类
    cd = CorrelationDiversifier(returns_df)
    selected = cd.select_portfolio(scored_df, n=5, method="greedy")
    dr = cd.get_diversification_ratio(selected)
    print(f"\n[5] Diversification Ratio: {dr:.4f}")
    selected2 = cd.select_portfolio(scored_df, n=5, method="score_penalty")
    print(f"    Score Penalty selection: {selected2}")

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
