"""Distance Correlation — Nonlinear Dependence Measurement
对应课程：Project 2 Feature Selection (distance correlation)

Distance Correlation 比 Pearson Correlation 更适合捕捉因子与收益之间的
非线性关系。在 Mini-GRP 中可用于：
    1. 替代 Pearson 做 Rank IC 计算
    2. 特征选择（筛选与收益有非线性依赖的因子）
    3. 因子间共线性检测（非线性版本）

Distance Correlation 定义（Székely et al., 2007）:
    - dCov²(X,Y) = E[|X-X'||Y-Y'|] + E[|X-X'|]E[|Y-Y'|] - 2E[|X-X'||Y-Y''|]
    - dCor(X,Y) = dCov(X,Y) / sqrt(dCov(X,X) * dCov(Y,Y))
    - dCor = 0  iff  X and Y are independent（比 Pearson 更强）

实现特点：
    - 纯 numpy 实现，不依赖 dcor 库
    - 使用 double-centering（U-centering）算法
    - 数值稳定，支持大规模数据

References:
    - Székely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007).
      "Measuring and testing dependence by correlation of distances".
      Annals of Statistics, 35(6): 2769-2794.
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd

__all__ = [
    "distance_covariance",
    "distance_correlation",
    "distance_correlation_matrix",
    "select_factors_by_distance_cor",
    "DistanceCorrelationAnalyzer",
]


def _double_center(dist_matrix: np.ndarray) -> np.ndarray:
    """
    对距离矩阵做 double-centering (U-centering)

    A_{ij} = a_{ij} - a_bar_{i.} - a_bar_{.j} + a_bar_{..}

    其中 a_bar_{i.} 是第 i 行的均值，a_bar_{.j} 是第 j 列的均值，
    a_bar_{..} 是全局均值。

    Parameters
    ----------
    dist_matrix : np.ndarray, shape (n, n)
        对称距离矩阵。

    Returns
    -------
    np.ndarray, shape (n, n)
        Double-centered 矩阵。
    """
    n = dist_matrix.shape[0]
    if n <= 1:
        return dist_matrix

    row_means = dist_matrix.mean(axis=1, keepdims=True)
    col_means = dist_matrix.mean(axis=0, keepdims=True)
    grand_mean = dist_matrix.mean()

    centered = dist_matrix - row_means - col_means + grand_mean
    return centered


def distance_covariance(x: np.ndarray, y: np.ndarray) -> float:
    """
    计算 distance covariance (dCov²)

    算法步骤：
        1. 计算 x 和 y 的 pairwise Euclidean distance matrices
        2. 对两个距离矩阵分别做 double-centering
        3. dCov² = mean(A * B)，其中 * 是逐元素乘法

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        一维数组，样本量 n。
    y : np.ndarray, shape (n,)
        一维数组，样本量 n。

    Returns
    -------
    float
        Distance covariance squared (dCov²)。
        如果输入长度不一致或长度 < 2，返回 0.0。

    Notes
    -----
    - 纯 numpy 实现，不依赖任何外部库
    - 数值稳定：使用 double precision
    - 时间复杂度 O(n²)，空间复杂度 O(n²)
    - 对于 n > 10000 的大规模数据，建议抽样

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> dCov2 = distance_covariance(x, y)
    >>> print(f"dCov²(x, x) = {dCov2:.6f}")
    dCov²(x, x) > 0
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"x and y must have same length, got {x.shape[0]} and {y.shape[0]}"
        )

    n = x.shape[0]
    if n < 2:
        return 0.0

    # 去NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    n_clean = x_clean.shape[0]

    if n_clean < 2:
        return 0.0

    # 步骤1: Pairwise distance matrices
    dx = np.abs(x_clean[:, None] - x_clean[None, :])
    dy = np.abs(y_clean[:, None] - y_clean[None, :])

    # 步骤2: Double-centering
    A = _double_center(dx)
    B = _double_center(dy)

    # 步骤3: dCov² = mean(A * B)
    dCov_sq = float(np.mean(A * B))

    # 非负修正（数值误差可能导致极小负值）
    return max(dCov_sq, 0.0)


def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Distance Correlation = dCov(x,y) / sqrt(dCov(x,x) * dCov(y,y))

    范围: [0, 1]
    dCor = 0  iff  x and y are independent

    相比 Pearson 的优势：
        - 能捕捉非线性关系
        - 对单调变换不变
        - dCor = 0 等价于独立（Pearson = 0 不等价于独立）

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        一维数组。
    y : np.ndarray, shape (n,)
        一维数组。

    Returns
    -------
    float
        Distance Correlation，范围 [0, 1]。

    Examples
    --------
    >>> # 线性关系
    >>> x = np.random.randn(100)
    >>> y = 2 * x + 1
    >>> dCor = distance_correlation(x, y)
    >>> print(f"dCor(linear) = {dCor:.4f}")  # ≈ 1.0

    >>> # 非线性关系（Pearson 无法捕捉）
    >>> y_nl = x ** 2
    >>> dCor_nl = distance_correlation(x, y_nl)
    >>> print(f"dCor(quadratic) = {dCor_nl:.4f}")  # 显著 > 0
    """
    dCov_xy = distance_covariance(x, y)
    dCov_xx = distance_covariance(x, x)
    dCov_yy = distance_covariance(y, y)

    denom = dCov_xx * dCov_yy

    if denom < 1e-15:
        return 0.0

    dCor = np.sqrt(dCov_xy) / np.sqrt(np.sqrt(denom))
    # 数值修正
    return float(min(max(dCor, 0.0), 1.0))


def distance_correlation_matrix(
    factor_df: pd.DataFrame,
    factor_cols: List[str],
    target: pd.Series,
) -> pd.DataFrame:
    """
    计算每个因子与 target 的 Distance Correlation + Pearson Correlation

    Parameters
    ----------
    factor_df : pd.DataFrame
        因子数据矩阵。
    factor_cols : List[str]
        因子列名列表。
    target : pd.Series
        目标变量（如下期收益率），index 与 factor_df 对齐。

    Returns
    -------
    pd.DataFrame
        columns:
            - factor: 因子名
            - pearson_ic: Pearson correlation
            - distance_ic: Distance Correlation
            - nonlinear_boost: distance_ic - pearson_ic

    nonlinear_boost 解读：
        - > 0.02: 该因子有显著的非线性成分，建议保留
        - < -0.02: Distance Correlation 显著低于 Pearson，可能有异常值影响
        - ~0: 关系基本线性，Pearson 已足够

    Examples
    --------
    >>> result = distance_correlation_matrix(factor_df, factor_cols, target_series)
    >>> print(result.sort_values('distance_ic', ascending=False).head())
    """
    results = []

    for col in factor_cols:
        if col not in factor_df.columns:
            continue

        # 对齐数据
        aligned = pd.concat([factor_df[col], target], axis=1).dropna()
        if len(aligned) < 3:
            continue

        x = aligned.iloc[:, 0].values
        y = aligned.iloc[:, 1].values

        # Pearson IC
        pearson_ic = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else 0.0
        if np.isnan(pearson_ic):
            pearson_ic = 0.0

        # Distance IC
        distance_ic = distance_correlation(x, y)

        # Nonlinear boost
        nonlinear_boost = distance_ic - abs(pearson_ic)

        results.append(
            {
                "factor": col,
                "pearson_ic": round(pearson_ic, 6),
                "distance_ic": round(distance_ic, 6),
                "nonlinear_boost": round(nonlinear_boost, 6),
            }
        )

    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        result_df = result_df.sort_values("distance_ic", ascending=False)

    return result_df


def select_factors_by_distance_cor(
    factor_df: pd.DataFrame,
    factor_cols: List[str],
    target: pd.Series,
    top_n: int = 10,
) -> List[str]:
    """
    按 Distance Correlation 排序，选出 Top N 因子

    Parameters
    ----------
    factor_df : pd.DataFrame
        因子数据矩阵。
    factor_cols : List[str]
        因子列名列表。
    target : pd.Series
        目标变量。
    top_n : int, default 10
        选取的因子数量。

    Returns
    -------
    List[str]
        Top N 因子名列表（按 Distance Correlation 降序）。

    Examples
    --------
    >>> top_factors = select_factors_by_distance_cor(factor_df, cols, target, top_n=5)
    >>> print(top_factors)
    ['momentum_20d', 'volatility_60d', 'rsi_14', 'volume_ratio', 'macd_signal']
    """
    result = distance_correlation_matrix(factor_df, factor_cols, target)
    if len(result) == 0:
        return []

    top = result.nlargest(top_n, "distance_ic")
    return top["factor"].tolist()


class DistanceCorrelationAnalyzer:
    """
    Distance Correlation 分析统一接口

    封装因子 Distance Correlation 分析的全流程，包括：
    - 计算 Distance Correlation 矩阵
    - 识别非线性因子
    - 生成对比报告

    Parameters
    ----------
    factor_df : pd.DataFrame
        因子数据矩阵。
    factor_cols : List[str]
        因子列名列表。
    target : pd.Series
        目标变量（如下期收益率）。

    Attributes
    ----------
    result : pd.DataFrame
        分析结果（调用 analyze_all() 后填充）。

    Examples
    --------
    >>> analyzer = DistanceCorrelationAnalyzer(factor_df, factor_cols, target)
    >>> result = analyzer.analyze_all()
    >>> nonlinear_factors = analyzer.get_nonlinear_factors(threshold=0.02)
    >>> report = analyzer.get_comparison_report()
    >>> print(report)
    """

    def __init__(
        self,
        factor_df: pd.DataFrame,
        factor_cols: List[str],
        target: pd.Series,
    ):
        self.factor_df = factor_df
        self.factor_cols = [c for c in factor_cols if c in factor_df.columns]
        self.target = target
        self._result: Optional[pd.DataFrame] = None

    def analyze_all(self) -> pd.DataFrame:
        """
        执行完整的 Distance Correlation 分析

        Returns
        -------
        pd.DataFrame
            columns: factor | pearson_ic | distance_ic | nonlinear_boost
            按 distance_ic 降序排列。
        """
        self._result = distance_correlation_matrix(
            self.factor_df, self.factor_cols, self.target
        )
        return self._result.copy()

    def get_nonlinear_factors(self, threshold: float = 0.02) -> List[str]:
        """
        返回 distance_ic - abs(pearson_ic) > threshold 的因子
        （有显著非线性成分的因子）

        Parameters
        ----------
        threshold : float, default 0.02
            非线性提升阈值。

        Returns
        -------
        List[str]
            具有显著非线性成分的因子名列表。
        """
        if self._result is None:
            self.analyze_all()

        assert self._result is not None
        mask = self._result["nonlinear_boost"] > threshold
        return self._result[mask]["factor"].tolist()

    def get_comparison_report(self) -> Dict:
        """
        Pearson vs Distance Correlation 对比报告

        Returns
        -------
        Dict
            包含以下键：
            - n_factors: 分析的总因子数
            - top_pearson: Pearson IC 最高的5个因子
            - top_distance: Distance IC 最高的5个因子
            - nonlinear_factors: 有显著非线性成分的因子
            - pearson_only: 只在 Pearson 中排名高
            - distance_only: 只在 Distance Correlation 中排名高
            - summary: 文字摘要

        Examples
        --------
        >>> report = analyzer.get_comparison_report()
        >>> print(report['summary'])
        """
        if self._result is None:
            self.analyze_all()

        assert self._result is not None
        df = self._result.copy()

        # 计算排名
        df["pearson_rank"] = df["pearson_ic"].abs().rank(ascending=False)
        df["distance_rank"] = df["distance_ic"].rank(ascending=False)

        n = len(df)

        top_pearson = df.nsmallest(5, "pearson_rank")[["factor", "pearson_ic"]].to_dict(
            "records"
        )
        top_distance = df.nsmallest(5, "distance_rank")[
            ["factor", "distance_ic"]
        ].to_dict("records")

        # 只在 Distance 中排名高（distance_rank 高但 pearson_rank 低）
        distance_only = df[
            (df["distance_rank"] <= 10) & (df["pearson_rank"] > 10)
        ]["factor"].tolist()

        # 只在 Pearson 中排名高
        pearson_only = df[
            (df["pearson_rank"] <= 10) & (df["distance_rank"] > 10)
        ]["factor"].tolist()

        nonlinear = df[df["nonlinear_boost"] > 0.02]["factor"].tolist()

        summary = (
            f"分析了 {n} 个因子。"
            f"Distance Correlation 排名前5的因子: {[d['factor'] for d in top_distance]}. "
            f"其中 {len(nonlinear)} 个因子有显著非线性成分。"
        )

        return {
            "n_factors": n,
            "top_pearson": top_pearson,
            "top_distance": top_distance,
            "nonlinear_factors": nonlinear,
            "pearson_only": pearson_only,
            "distance_only": distance_only,
            "summary": summary,
        }


# ───────────────────────────────────────────────────────────────────────────────
# 用法示例（可独立运行）
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Demo: Distance Correlation")
    print("=" * 60)

    np.random.seed(42)

    # 1) 基本 Distance Covariance / Correlation
    n = 200
    x = np.random.randn(n)

    # 线性关系
    y_linear = 2 * x + 1 + np.random.randn(n) * 0.1
    dCov_lin = distance_covariance(x, y_linear)
    dCor_lin = distance_correlation(x, y_linear)
    pearson_lin = np.corrcoef(x, y_linear)[0, 1]
    print(f"\n[1] Linear relationship: y = 2x + 1 + noise")
    print(f"    Pearson = {pearson_lin:.4f}")
    print(f"    dCov²   = {dCov_lin:.6f}")
    print(f"    dCor    = {dCor_lin:.4f}")

    # 二次非线性关系（Pearson 应该接近 0，但 dCor 应该显著 > 0）
    y_quad = x ** 2 + np.random.randn(n) * 0.1
    dCov_quad = distance_covariance(x, y_quad)
    dCor_quad = distance_correlation(x, y_quad)
    pearson_quad = np.corrcoef(x, y_quad)[0, 1]
    print(f"\n[2] Quadratic relationship: y = x² + noise")
    print(f"    Pearson = {pearson_quad:.4f}  (≈ 0, 无法捕捉)")
    print(f"    dCov²   = {dCov_quad:.6f}")
    print(f"    dCor    = {dCor_quad:.4f}  (> 0, 成功捕捉非线性)")

    # 2) 因子 Distance Correlation 矩阵
    print(f"\n[3] Factor Distance Correlation Matrix:")
    n_days = 100
    n_factors = 8
    factor_names = [f"factor_{i}" for i in range(n_factors)]

    # 构造因子数据：部分与 target 线性相关，部分非线性相关
    factor_data = pd.DataFrame(index=range(n_days))
    base = np.random.randn(n_days)

    factor_data["momentum_lin"] = 0.5 * base + np.random.randn(n_days) * 0.3
    factor_data["value_lin"] = -0.3 * base + np.random.randn(n_days) * 0.5
    factor_data["volatility_nl"] = np.abs(base) + np.random.randn(n_days) * 0.2
    factor_data["rsi_nl"] = base ** 2 + np.random.randn(n_days) * 0.3
    factor_data["macd_noise"] = np.random.randn(n_days)
    factor_data["volume_noise"] = np.random.randn(n_days)
    factor_data["trend_lin"] = 0.4 * base + np.random.randn(n_days) * 0.4
    factor_data["reversal_nl"] = np.sin(base * 2) + np.random.randn(n_days) * 0.2

    target_series = 0.3 * base + 0.2 * base ** 2 + np.random.randn(n_days) * 0.2

    result = distance_correlation_matrix(
        factor_data, factor_data.columns.tolist(), pd.Series(target_series)
    )
    print(result.to_string(index=False))

    # 3) Top N 因子选择
    top_factors = select_factors_by_distance_cor(
        factor_data, factor_data.columns.tolist(), pd.Series(target_series), top_n=5
    )
    print(f"\n[4] Top 5 factors by Distance Correlation: {top_factors}")

    # 4) DistanceCorrelationAnalyzer 类
    analyzer = DistanceCorrelationAnalyzer(
        factor_data, factor_data.columns.tolist(), pd.Series(target_series)
    )
    report = analyzer.get_comparison_report()
    print(f"\n[5] Comparison Report:")
    print(f"    {report['summary']}")
    print(f"    Nonlinear factors: {report['nonlinear_factors']}")
    print(f"    Distance-only factors: {report['distance_only']}")

    # 验证关键性质：dCor(x, x) == 1
    self_dcor = distance_correlation(x, x)
    print(f"\n[6] dCor(x, x) = {self_dcor:.6f} (should be ≈ 1.0)")

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
