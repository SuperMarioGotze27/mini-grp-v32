"""
Monte Carlo Portfolio Risk Simulation
======================================
对应课程：Lecture 6-7 (Monte Carlo Simulation)

核心应用：
1. Parametric MC: 假设收益率服从multivariate normal
2. Historical Bootstrap: 从历史收益率中block bootstrap
3. VaR/CVaR计算
4. 多horizon模拟（1M/3M/6M/12M）
5. 风险分解（哪些因子贡献最大风险）

关键概念（Lecture 6-7）：
- Monte Carlo Simulation: 通过大量随机路径模拟资产未来价格/收益分布
- Law of Large Numbers: 模拟次数越多，估计越精确
- Jensen's Inequality: E[f(X)] >= f(E[X])，解释为何parametric与bootstrap结果可能不同
- Block Bootstrap: 保留时间序列局部相关性，对应Lecture 6的serial correlation处理
- VaR: Value at Risk, 给定置信水平下的最大损失
- CVaR/Expected Shortfall: 超过VaR阈值后的条件期望损失
- Factor Risk Decomposition: 将总风险归因到各因子维度

Author: Mini-GRP Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

__all__ = [
    "monte_carlo_parametric",
    "monte_carlo_bootstrap",
    "var_cvar",
    "factor_risk_decomposition",
    "MonteCarloRiskEngine",
]

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _validate_inputs(returns: pd.DataFrame, weights: np.ndarray) -> None:
    """
    校验输入数据的一致性和合法性。

    对应Lecture 6: 输入质量是MC结果可靠性的前提。
    """
    if returns.empty:
        raise ValueError("returns DataFrame cannot be empty")
    if len(weights) != returns.shape[1]:
        raise ValueError(
            f"weights length ({len(weights)}) must match number of stocks "
            f"({returns.shape[1]})"
        )
    if not np.isclose(np.sum(weights), 1.0, atol=1e-6):
        raise ValueError(f"weights must sum to 1, got {np.sum(weights):.6f}")


def _compute_summary(simulated_returns: np.ndarray, annualize: bool = False,
                     horizon: int = 63, trading_days: int = 252) -> Dict:
    """
    从模拟收益分布中抽取全部风险统计量。

    Parameters
    ----------
    simulated_returns : np.ndarray, shape (n_sims,)
        单期累积收益率（已加1复利后减1，或直接累积对数收益后exp-1）
    annualize : bool
        是否将统计量年化
    horizon : int
        模拟持有期（交易日）
    trading_days : int
        年化基准（默认252个交易日）

    Returns
    -------
    dict : 包含全部风险指标的字典
    """
    mean_ret = float(np.mean(simulated_returns))
    vol_ret = float(np.std(simulated_returns, ddof=1))

    var_95, var_99, cvar_95, cvar_99 = var_cvar(simulated_returns)

    prob_positive = float(np.mean(simulated_returns > 0))
    prob_beat_benchmark = float(np.mean(simulated_returns > 0.05))

    worst_case = float(np.percentile(simulated_returns, 1))
    best_case = float(np.percentile(simulated_returns, 99))

    pct_levels = [5, 10, 25, 50, 75, 90, 95]
    percentiles = {int(p): float(np.percentile(simulated_returns, p))
                   for p in pct_levels}

    # ---- 年化处理 ----
    if annualize:
        ann_factor = trading_days / horizon
        mean_ret = mean_ret * ann_factor
        vol_ret = vol_ret * np.sqrt(ann_factor)
        var_95 = var_95 * ann_factor
        var_99 = var_99 * ann_factor
        cvar_95 = cvar_95 * ann_factor
        cvar_99 = cvar_99 * ann_factor
        worst_case = worst_case * ann_factor
        best_case = best_case * ann_factor
        percentiles = {k: v * ann_factor for k, v in percentiles.items()}

    return {
        "simulated_returns": simulated_returns,
        "expected_return": mean_ret,
        "volatility": vol_ret,
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "cvar_99": cvar_99,
        "prob_positive": prob_positive,
        "prob_beat_benchmark": prob_beat_benchmark,
        "worst_case": worst_case,
        "best_case": best_case,
        "percentiles": percentiles,
    }


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------

def monte_carlo_parametric(
    returns: pd.DataFrame,
    weights: np.ndarray,
    horizon: int = 63,
    n_sims: int = 10000,
    annualize: bool = True,
) -> Dict:
    """
    Parametric Monte Carlo simulation assuming multivariate normal.

    对应Lecture 6: "Monte Carlo with Parametric Assumptions"
    - 假设日收益率服从多元正态分布 N(μ, Σ)
    - 利用Cholesky分解生成相关随机数
    - 对应课程中的Geometric Brownian Motion (GBM) 离散化思想

    Parameters
    ----------
    returns : pd.DataFrame, shape (n_days, n_stocks)
        历史日收益率矩阵，每列为一只股票。
    weights : np.ndarray, shape (n_stocks,)
        组合权重，必须和为1。
    horizon : int, default 63
        模拟持有期（交易日数）。63 ≈ 3个月。
    n_sims : int, default 10000
        Monte Carlo 模拟路径数。对应Law of Large Numbers:
        路径越多，尾部估计越稳定。
    annualize : bool, default True
        是否将结果年化。年化期望收益 = 原始 × (252/horizon)，
        年化波动率 = 原始 × sqrt(252/horizon)。

    Returns
    -------
    dict
        {
            'simulated_returns': np.ndarray, shape (n_sims,)
                期末累积收益率分布（不加复利时的简单累积）
            'expected_return': float
                模拟分布的均值
            'volatility': float
                模拟分布的标准差
            'var_95': float
                5% VaR（负值表示损失）
            'var_99': float
                1% VaR
            'cvar_95': float
                5% CVaR / Expected Shortfall
            'cvar_99': float
                1% CVaR
            'prob_positive': float
                P(累积收益 > 0)
            'prob_beat_benchmark': float
                P(累积收益 > 5%)
            'worst_case': float
                1% percentile（最差情景）
            'best_case': float
                99% percentile（最好情景）
            'percentiles': dict
                {5: ..., 10: ..., 25: ..., 50: ..., 75: ..., 90: ..., 95: ...}
        }
    """
    _validate_inputs(returns, weights)

    mu = returns.mean().values              # (n_stocks,) 日均期望收益
    Sigma = returns.cov().values            # (n_stocks, n_stocks) 协方差
    n_stocks = returns.shape[1]

    # Cholesky分解 — 对应Lecture 6 "Generating Correlated Random Variables"
    # L @ L.T = Sigma，用于将独立标准正态变换为相关正态
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        # 若协方差矩阵非正定，添加微小扰动后重试
        Sigma += np.eye(n_stocks) * 1e-8
        L = np.linalg.cholesky(Sigma)

    # 生成独立标准正态随机数 -> Cholesky变换 -> 相关随机数
    # 对应Lecture 6: "From Uncorrelated to Correlated Normals"
    Z = np.random.standard_normal((horizon, n_sims, n_stocks))
    # 将 Z reshape 为 (horizon*n_sims, n_stocks)，批量做 L.T @ z' 变换
    Z_flat = Z.reshape(-1, n_stocks)          # (horizon*n_sims, n_stocks)
    daily_returns_flat = Z_flat @ L.T          # (horizon*n_sims, n_stocks)
    daily_returns = daily_returns_flat.reshape(horizon, n_sims, n_stocks)
    daily_returns += mu[None, None, :]         # 加上均值
    # shape: (horizon, n_sims, n_stocks)

    # 组合日收益 = 各股票日收益 × 权重
    portfolio_daily = np.einsum("tji,i->tj", daily_returns, weights)
    # shape: (horizon, n_sims)

    # 累积收益（简单加总）— 对应Lecture 6离散时间累积
    cumulative_returns = np.sum(portfolio_daily, axis=0)  # (n_sims,)

    return _compute_summary(cumulative_returns, annualize=annualize, horizon=horizon)


def monte_carlo_bootstrap(
    returns: pd.DataFrame,
    weights: np.ndarray,
    horizon: int = 63,
    n_sims: int = 10000,
    block_size: int = 5,
) -> Dict:
    """
    Historical Bootstrap MC — 不假设正态分布，从历史中 block bootstrap。

    对应Lecture 6-7: "Non-parametric / Bootstrap Monte Carlo"
    - 直接从历史数据中抽样，不假设任何分布
    - Block Bootstrap: 以块为单位抽样，保留局部时间序列相关性
    - 对应Jensen's Inequality讨论: bootstrap保留历史偏度和峰度

    Parameters
    ----------
    returns : pd.DataFrame, shape (n_days, n_stocks)
        历史日收益率矩阵。
    weights : np.ndarray, shape (n_stocks,)
        组合权重，必须和为1。
    horizon : int, default 63
        模拟持有期（交易日数）。
    n_sims : int, default 10000
        Monte Carlo 模拟路径数。
    block_size : int, default 5
        Block bootstrap 块大小。块越大，保留的自相关性越强，
        但有效样本量越小。对应Lecture 6处理serial correlation的方法。

    Returns
    -------
    dict
        返回结构与 monte_carlo_parametric 完全一致。
    """
    _validate_inputs(returns, weights)

    returns_arr = returns.values  # (n_days, n_stocks)
    n_days, n_stocks = returns_arr.shape

    if n_days < block_size:
        raise ValueError(
            f"block_size ({block_size}) cannot exceed available history ({n_days})"
        )

    # 可抽取的block数量
    n_blocks = n_days - block_size + 1
    # 每个模拟路径需要的block数
    blocks_per_path = int(np.ceil(horizon / block_size))
    # 每个block的起始索引
    block_starts = np.arange(n_blocks)

    simulated_returns = np.zeros(n_sims)

    for sim in range(n_sims):
        # 随机抽取blocks
        chosen_starts = np.random.choice(block_starts, size=blocks_per_path, replace=True)

        # 拼接block
        path_returns = []
        for start in chosen_starts:
            block = returns_arr[start : start + block_size]  # (block_size, n_stocks)
            path_returns.append(block)
            if len(path_returns) * block_size >= horizon:
                break

        path_returns = np.concatenate(path_returns, axis=0)[:horizon]  # (horizon, n_stocks)
        portfolio_daily = path_returns @ weights  # (horizon,)
        simulated_returns[sim] = np.sum(portfolio_daily)

    # bootstrap结果不annualize（因为是从原始日频数据直接抽样）
    return _compute_summary(simulated_returns, annualize=False, horizon=horizon)


def var_cvar(simulated_returns: np.ndarray) -> Tuple[float, float, float, float]:
    """
    计算 VaR 和 CVaR（Conditional Value at Risk / Expected Shortfall）。

    对应Lecture 7: "Risk Measures — VaR and Expected Shortfall"
    - VaR: 给定置信水平α下的分位数损失
    - CVaR/ES: 超过VaR阈值后的条件平均损失
    - CVaR满足一致性风险度量(coherent risk measure)的四条公理

    Parameters
    ----------
    simulated_returns : np.ndarray, shape (n_sims,)
        Monte Carlo 模拟得到的期末累积收益率分布。

    Returns
    -------
    tuple
        (var_95, var_99, cvar_95, cvar_99)
        - var_95: 5% VaR（95%置信水平），例如 -0.05 表示5%概率损失超过5%
        - var_99: 1% VaR（99%置信水平）
        - cvar_95: 5% CVaR，即收益 < var_95 时的条件期望
        - cvar_99: 1% CVaR
    """
    if simulated_returns.size == 0:
        raise ValueError("simulated_returns cannot be empty")

    var_95 = float(np.percentile(simulated_returns, 5))
    var_99 = float(np.percentile(simulated_returns, 1))

    # CVaR = 低于VaR阈值的所有模拟值的平均
    cvar_95 = float(np.mean(simulated_returns[simulated_returns <= var_95]))
    cvar_99 = float(np.mean(simulated_returns[simulated_returns <= var_99]))

    return var_95, var_99, cvar_95, cvar_99


def factor_risk_decomposition(
    returns: pd.DataFrame,
    weights: np.ndarray,
    factor_exposures: pd.DataFrame,
) -> pd.DataFrame:
    """
    风险归因：将组合风险分解到各因子维度。

    对应Lecture 7: "Factor Risk Decomposition"
    - 总风险（方差）可分解为各因子的边际贡献
    - 基于Marginal Contribution to Risk (MCTR) 框架
    - 组合方差 = w'Σw，其中Σ为因子协方差矩阵
    - 各因子贡献 = (B w)'_k · (Σ B w)_k / σ_p
      其中 B 为因子暴露矩阵，Σ 为因子收益协方差

    Parameters
    ----------
    returns : pd.DataFrame, shape (n_days, n_stocks)
        历史日收益率。
    weights : np.ndarray, shape (n_stocks,)
        组合权重。
    factor_exposures : pd.DataFrame, shape (n_stocks, n_factors)
        每只股票在各因子上的暴露。
        例如 factor_exposures.columns = ['Value', 'Quality', 'Growth', 'Momentum']

    Returns
    -------
    pd.DataFrame
        列: factor | contribution | pct_contribution
        - factor: 因子名称
        - contribution: 该因子对组合波动率的绝对贡献
        - pct_contribution: 该因子的贡献占比（百分比）
    """
    _validate_inputs(returns, weights)

    if factor_exposures.shape[0] != returns.shape[1]:
        raise ValueError(
            f"factor_exposures rows ({factor_exposures.shape[0]}) must match "
            f"number of stocks ({returns.shape[1]})"
        )

    factor_names = list(factor_exposures.columns)
    B = factor_exposures.values  # (n_stocks, n_factors)

    # 组合因子暴露 = B' @ weights
    # 对应Lecture 7: "Portfolio Factor Exposure = sum(w_i * beta_i_k)"
    portfolio_exposure = B.T @ weights  # (n_factors,)

    # 计算因子收益: 因子收益 = 股票收益的横截面回归系数
    # 使用OLS: r = B @ f + ε  =>  f ≈ (B'B)^{-1} B' r
    # 这里采用简化方法：因子收益协方差矩阵
    try:
        BtB_inv = np.linalg.inv(B.T @ B)
    except np.linalg.LinAlgError:
        BtB_inv = np.linalg.pinv(B.T @ B)

    # 因子收益时间序列: 每日对每只股票的因子收益
    factor_returns = returns.values @ B @ BtB_inv  # (n_days, n_factors)
    factor_cov = np.cov(factor_returns, rowvar=False)  # (n_factors, n_factors)

    # 组合在各因子上的边际风险贡献 (MCTR)
    # σ_p = sqrt(w'B Σ_f B'w)
    # MCTR_k = (B'w)_k · (Σ_f B'w)_k / σ_p
    factor_vol_vector = factor_cov @ portfolio_exposure  # (n_factors,)
    total_portfolio_var = portfolio_exposure @ factor_cov @ portfolio_exposure

    if total_portfolio_var <= 0:
        # 退化情况：所有贡献均等
        contributions = np.ones(len(factor_names)) / len(factor_names)
        total_vol = 1.0
    else:
        total_vol = np.sqrt(total_portfolio_var)
        contributions = portfolio_exposure * factor_vol_vector / total_vol

    pct_contributions = 100.0 * contributions / np.sum(np.abs(contributions))

    result = pd.DataFrame({
        "factor": factor_names,
        "contribution": np.round(contributions, 6),
        "pct_contribution": np.round(pct_contributions, 2),
    })

    return result


# ---------------------------------------------------------------------------
# 统一引擎类
# ---------------------------------------------------------------------------

class MonteCarloRiskEngine:
    """
    MC风险模拟引擎 — 统一接口。

    对应Lecture 6-7的整体框架：提供一站式Monte Carlo风险分析能力。

    使用流程:
    --------
    1. engine = MonteCarloRiskEngine(returns_df, weights)
    2. result = engine.run(n_sims=10000, method='parametric')
    3. var_95 = result['var_95']
    4. plot_data = engine.get_distribution_plot_data()
    5. multi_horizon = engine.multi_horizon_analysis()

    Parameters
    ----------
    returns : pd.DataFrame
        历史日收益率矩阵 (n_days × n_stocks)。
    weights : np.ndarray
        组合权重 (n_stocks,)，必须和为1。
    """

    def __init__(self, returns: pd.DataFrame, weights: np.ndarray):
        """
        初始化MC风险引擎。

        对应Lecture 6: 引擎初始化时对输入数据进行校验和预处理。
        """
        _validate_inputs(returns, weights)
        self.returns = returns
        self.weights = np.asarray(weights, dtype=float)
        self._last_result: Optional[Dict] = None
        self._last_method: Optional[str] = None

    # ------------------------------------------------------------------ #

    def run(
        self,
        n_sims: int = 10000,
        horizon: int = 63,
        method: str = "parametric",
        **kwargs,
    ) -> Dict:
        """
        运行MC simulation。

        Parameters
        ----------
        n_sims : int, default 10000
            模拟次数。
        horizon : int, default 63
            持有期（交易日）。63 ≈ 3个月，126 ≈ 6个月，252 ≈ 1年。
        method : str, default 'parametric'
            模拟方法，'parametric' 或 'bootstrap'。
        **kwargs
            额外参数传递给底层函数（如 block_size for bootstrap）。

        Returns
        -------
        dict
            同 monte_carlo_parametric / monte_carlo_bootstrap 的返回。
        """
        if method not in ("parametric", "bootstrap"):
            raise ValueError(f"method must be 'parametric' or 'bootstrap', got '{method}'")

        if method == "parametric":
            annualize = kwargs.get("annualize", True)
            result = monte_carlo_parametric(
                self.returns, self.weights,
                horizon=horizon, n_sims=n_sims, annualize=annualize,
            )
        else:
            block_size = kwargs.get("block_size", 5)
            result = monte_carlo_bootstrap(
                self.returns, self.weights,
                horizon=horizon, n_sims=n_sims, block_size=block_size,
            )

        self._last_result = result
        self._last_method = method
        return result

    # ------------------------------------------------------------------ #

    def get_distribution_plot_data(self) -> Dict:
        """
        返回前端绘图数据，用于直方图可视化模拟收益分布。

        对应Lecture 6-7: 可视化是MC分析的重要环节，帮助直观理解分布形态。

        Returns
        -------
        dict
            {
                'bins': list of float
                    直方图bin边界
                'hist': list of int
                    每个bin的频数
                'var_95': float
                    5% VaR位置
                'cvar_95': float
                    5% CVaR位置
                'mean': float
                    分布均值
                'median': float
                    分布中位数
            }

        Raises
        ------
        RuntimeError
            若尚未调用 run() 方法。
        """
        if self._last_result is None:
            raise RuntimeError("Must call run() before get_distribution_plot_data()")

        sim_returns = self._last_result["simulated_returns"]

        # 使用50个bin构建直方图
        counts, bin_edges = np.histogram(sim_returns, bins=50)

        return {
            "bins": bin_edges.tolist(),
            "hist": counts.tolist(),
            "var_95": self._last_result["var_95"],
            "cvar_95": self._last_result["cvar_95"],
            "mean": self._last_result["expected_return"],
            "median": self._last_result["percentiles"][50],
        }

    # ------------------------------------------------------------------ #

    def multi_horizon_analysis(
        self,
        horizons: Optional[List[int]] = None,
        n_sims: int = 10000,
        method: str = "parametric",
        **kwargs,
    ) -> pd.DataFrame:
        """
        多期限风险分析 — 同时分析多个持有期的风险指标。

        对应Lecture 7: "Multi-horizon Risk Assessment"
        - 短期风险（1个月）vs 长期风险（1年）
        - 波动率随时间的平方根缩放: σ_T = σ_1 × sqrt(T)
        - 但VaR不线性缩放，需通过MC直接模拟

        Parameters
        ----------
        horizons : list of int, optional
            持有期列表（交易日）。默认 [21, 63, 126, 252]
            分别对应约 1个月 / 3个月 / 6个月 / 1年。
        n_sims : int, default 10000
            每个horizon的模拟次数。
        method : str, default 'parametric'
            模拟方法。
        **kwargs
            额外参数传递给底层函数。

        Returns
        -------
        pd.DataFrame
            列: horizon | expected_return | vol | var_95 | cvar_95
                - horizon: 持有期（交易日）
                - expected_return: 年化期望收益
                - vol: 年化波动率
                - var_95: 年化5% VaR
                - cvar_95: 年化5% CVaR
        """
        if horizons is None:
            horizons = [21, 63, 126, 252]  # ~1M, 3M, 6M, 1Y

        records = []
        for h in horizons:
            res = self.run(n_sims=n_sims, horizon=h, method=method, **kwargs)
            records.append({
                "horizon": h,
                "expected_return": res["expected_return"],
                "vol": res["volatility"],
                "var_95": res["var_95"],
                "cvar_95": res["cvar_95"],
            })

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 用法示例 (module 被直接运行时执行)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("Monte Carlo Portfolio Risk Simulation — Demo")
    print("=" * 60)

    # ---- 1. 生成模拟数据: 5只股票，500个交易日 ----
    n_days = 500
    n_stocks = 5
    stock_names = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    # 构建相关性结构
    corr = np.array([
        [1.00, 0.65, 0.55, 0.50, 0.40],
        [0.65, 1.00, 0.60, 0.45, 0.35],
        [0.55, 0.60, 1.00, 0.50, 0.45],
        [0.50, 0.45, 0.50, 1.00, 0.30],
        [0.40, 0.35, 0.45, 0.30, 1.00],
    ])
    vols = np.array([0.016, 0.014, 0.018, 0.020, 0.025])  # 日波动率
    Sigma = np.diag(vols) @ corr @ np.diag(vols)
    mu = np.array([0.0006, 0.0005, 0.0007, 0.0004, 0.0008])  # 日期望收益

    L = np.linalg.cholesky(Sigma)
    raw_returns = np.random.standard_normal((n_days, n_stocks))
    returns_arr = raw_returns @ L.T + mu
    returns_df = pd.DataFrame(returns_arr, columns=stock_names)

    # 等权重组合
    weights = np.array([0.20, 0.20, 0.20, 0.20, 0.20])

    print(f"\n[数据] {n_stocks} 只股票, {n_days} 个交易日")
    print(f"[权重] 等权重: {weights}")

    # ---- 2. Parametric MC ----
    print("\n" + "-" * 40)
    print("2. Parametric Monte Carlo (Multivariate Normal)")
    print("-" * 40)
    res_param = monte_carlo_parametric(
        returns_df, weights, horizon=63, n_sims=10000, annualize=True
    )
    print(f"  期望收益 (年化) : {res_param['expected_return']:.4f}")
    print(f"  波动率 (年化)   : {res_param['volatility']:.4f}")
    print(f"  VaR 95%         : {res_param['var_95']:.4f}")
    print(f"  VaR 99%         : {res_param['var_99']:.4f}")
    print(f"  CVaR 95%        : {res_param['cvar_95']:.4f}")
    print(f"  CVaR 99%        : {res_param['cvar_99']:.4f}")
    print(f"  P(Return>0)     : {res_param['prob_positive']:.4f}")
    print(f"  最差1%情景      : {res_param['worst_case']:.4f}")
    print(f"  最好1%情景      : {res_param['best_case']:.4f}")

    # ---- 3. Bootstrap MC ----
    print("\n" + "-" * 40)
    print("3. Historical Bootstrap MC (Block Size=5)")
    print("-" * 40)
    res_boot = monte_carlo_bootstrap(
        returns_df, weights, horizon=63, n_sims=10000, block_size=5
    )
    print(f"  期望收益        : {res_boot['expected_return']:.4f}")
    print(f"  波动率          : {res_boot['volatility']:.4f}")
    print(f"  VaR 95%         : {res_boot['var_95']:.4f}")
    print(f"  VaR 99%         : {res_boot['var_99']:.4f}")
    print(f"  CVaR 95%        : {res_boot['cvar_95']:.4f}")
    print(f"  CVaR 99%        : {res_boot['cvar_99']:.4f}")
    print(f"  P(Return>0)     : {res_boot['prob_positive']:.4f}")

    # ---- 4. VaR/CVaR 独立函数 ----
    print("\n" + "-" * 40)
    print("4. VaR/CVaR 计算")
    print("-" * 40)
    dummy = np.random.normal(0, 0.05, 10000)
    v95, v99, c95, c99 = var_cvar(dummy)
    print(f"  VaR 95% = {v95:.4f}, VaR 99% = {v99:.4f}")
    print(f"  CVaR 95% = {c95:.4f}, CVaR 99% = {c99:.4f}")

    # ---- 5. 因子风险分解 ----
    print("\n" + "-" * 40)
    print("5. Factor Risk Decomposition")
    print("-" * 40)
    factor_exp = pd.DataFrame(
        np.random.randn(n_stocks, 4),
        index=stock_names,
        columns=["Value", "Quality", "Growth", "Momentum"],
    )
    factor_result = factor_risk_decomposition(returns_df, weights, factor_exp)
    print(factor_result.to_string(index=False))

    # ---- 6. MonteCarloRiskEngine 统一接口 ----
    print("\n" + "-" * 40)
    print("6. MonteCarloRiskEngine — 统一接口")
    print("-" * 40)
    engine = MonteCarloRiskEngine(returns_df, weights)

    # 单次运行
    result = engine.run(n_sims=10000, horizon=63, method="parametric")
    print(f"  引擎运行完成: method=parametric, horizon=63")
    print(f"  VaR 95% = {result['var_95']:.4f}")

    # 绘图数据
    plot_data = engine.get_distribution_plot_data()
    print(f"  绘图数据: {len(plot_data['hist'])} bins")

    # 多期限分析
    print("\n  多期限分析:")
    multi = engine.multi_horizon_analysis(
        horizons=[21, 63, 126, 252], n_sims=5000, method="parametric"
    )
    print(multi.to_string(index=False))

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
