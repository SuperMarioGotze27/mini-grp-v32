"""
三层Regime Framework:
Layer 1: Rule-based Regime (可解释baseline)
Layer 2: HMM / Markov Switching (数据驱动)
Layer 3: Regime Conditional Factor IC验证 (核心)

依赖降级策略:
- hmmlearn: 未安装时用简化的高斯混合模型+EM实现
- scipy: 未安装时用numpy实现Welch's t-test简化版
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "RuleBasedRegime",
    "HMMRegimeDetector",
    "RegimeConditionalIC",
    "estimate_half_life",
]

# ---------------------------------------------------------------------------
# 依赖探测
# ---------------------------------------------------------------------------

_HMMLEARN_AVAILABLE = False
_SCIPY_AVAILABLE = False

try:
    from hmmlearn.hmm import GaussianHMM

    _HMMLEARN_AVAILABLE = True
except ImportError:
    warnings.warn("hmmlearn not available; HMMRegimeDetector uses numpy fallback.")

try:
    from scipy import stats

    _SCIPY_AVAILABLE = True
except ImportError:
    warnings.warn("scipy not available; t-tests use numpy fallback.")


# ---------------------------------------------------------------------------
# Layer 1: Rule-based Regime
# ---------------------------------------------------------------------------

class RuleBasedRegime:
    r"""
    Layer 1: 基于规则的regime检测 — 完全可解释的baseline。

    使用波动率、收益、流动性三个维度判断当前市场regime。
    规则优先级：Bear > Volatile > Bull > Range-Bound。

    Attributes
    ----------
    REGIME_RULES : dict
        各regime的判定阈值。
    """

    REGIME_RULES = {
        "Bull": {"return_20d_threshold": 0.05, "volatility_threshold": 0.15},
        "Bear": {"return_20d_threshold": -0.05, "volatility_threshold": 0.20},
        "Volatile": {"volatility_threshold": 0.25},
        "Range-Bound": {
            "volatility_threshold": 0.20,
            "return_range": (-0.05, 0.05),
        },
    }

    def detect(
        self,
        volatility: float,
        return_20d: float,
        liquidity: Optional[float] = None,
    ) -> str:
        r"""
        返回当前regime名称。

        判定逻辑（优先级从高到低）：
        1. Bear:   return_20d <= -0.05 且 volatility >= 0.20
        2. Volatile: volatility >= 0.25
        3. Bull:   return_20d >= 0.05  且 volatility <= 0.15
        4. Range-Bound: 以上均不满足

        Parameters
        ----------
        volatility : float
            年化波动率（如20日波动率年化）。
        return_20d : float
            近20日累计收益。
        liquidity : Optional[float]
            流动性指标（预留，当前不使用）。

        Returns
        -------
        str
            Regime名称: 'Bull' | 'Bear' | 'Volatile' | 'Range-Bound'。

        Examples
        --------
        >>> detector = RuleBasedRegime()
        >>> detector.detect(0.30, -0.08)
        'Bear'
        >>> detector.detect(0.10, 0.08)
        'Bull'
        """
        # Bear: 大幅下跌且高波动
        if (
            return_20d <= self.REGIME_RULES["Bear"]["return_20d_threshold"]
            and volatility >= self.REGIME_RULES["Bear"]["volatility_threshold"]
        ):
            return "Bear"

        # Volatile: 极高波动
        if volatility >= self.REGIME_RULES["Volatile"]["volatility_threshold"]:
            return "Volatile"

        # Bull: 上涨且低波动
        if (
            return_20d >= self.REGIME_RULES["Bull"]["return_20d_threshold"]
            and volatility <= self.REGIME_RULES["Bull"]["volatility_threshold"]
        ):
            return "Bull"

        # Range-Bound: 其他情况
        return "Range-Bound"

    def detect_series(
        self,
        volatility_series: pd.Series,
        return_series: pd.Series,
    ) -> pd.Series:
        """
        对时间序列批量检测regime。

        Parameters
        ----------
        volatility_series : pd.Series
            波动率时间序列。
        return_series : pd.Series
            收益时间序列。

        Returns
        -------
        pd.Series
            Regime时间序列。
        """
        regimes = []
        for vol, ret in zip(volatility_series, return_series):
            regimes.append(self.detect(vol, ret))
        return pd.Series(regimes, index=volatility_series.index, name="regime")


# ---------------------------------------------------------------------------
# Layer 2: HMM Regime Detector
# ---------------------------------------------------------------------------

class _NumpyGaussianHMM:
    r"""
    纯numpy实现的简化高斯HMM。

    使用EM算法估计参数。当hmmlearn不可用时作为fallback。
    仅支持高斯发射分布和对角协方差。

    Parameters
    ----------
    n_components : int
        隐状态（regime）数量。
    n_iter : int, default 100
        EM最大迭代次数。
    tol : float, default 1e-3
        收敛阈值。
    """

    def __init__(self, n_components: int = 3, n_iter: int = 100, tol: float = 1e-3):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.means_: Optional[np.ndarray] = None
        self.covars_: Optional[np.ndarray] = None
        self.transmat_: Optional[np.ndarray] = None
        self.startprob_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "_NumpyGaussianHMM":
        """使用EM算法训练HMM。"""
        n_samples, n_features = X.shape
        rng = np.random.RandomState(42)

        # 初始化参数
        self.startprob_ = np.ones(self.n_components) / self.n_components
        self.transmat_ = (
            np.eye(self.n_components) * 0.6
            + np.ones((self.n_components, self.n_components)) * 0.4 / self.n_components
        )
        # K-means初始化
        idx = rng.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[idx].copy()
        self.covars_ = np.array(
            [np.var(X, axis=0) + 1e-6 for _ in range(self.n_components)]
        )

        prev_loglik = -np.inf

        for _ in range(self.n_iter):
            # E-step: forward-backward
            log_alpha, log_beta, log_likelihood = self._forward_backward(X)

            # 计算gamma和xi
            log_gamma = log_alpha + log_beta - log_likelihood
            gamma = np.exp(log_gamma)

            # M-step: 更新参数
            # 更新startprob
            self.startprob_ = gamma[0] + 1e-8
            self.startprob_ /= self.startprob_.sum()

            # 更新means和covars
            for k in range(self.n_components):
                gamma_k = gamma[:, k : k + 1]
                sum_gamma = gamma_k.sum()
                if sum_gamma > 0:
                    self.means_[k] = (gamma_k * X).sum(axis=0) / sum_gamma
                    diff = X - self.means_[k]
                    self.covars_[k] = (gamma_k * (diff**2)).sum(axis=0) / sum_gamma + 1e-6

            # 更新transmat
            log_xi = self._compute_log_xi(X, log_alpha, log_beta, log_likelihood)
            xi = np.exp(log_xi)
            for i in range(self.n_components):
                denom = xi[:, i, :].sum()
                if denom > 0:
                    self.transmat_[i] = xi[:, i, :].sum(axis=0) / denom
                else:
                    self.transmat_[i] = np.ones(self.n_components) / self.n_components

            # 检查收敛
            if abs(log_likelihood - prev_loglik) < self.tol:
                break
            prev_loglik = log_likelihood

        return self

    def _forward_backward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """前向-后向算法。"""
        n_samples = len(X)
        log_alpha = np.zeros((n_samples, self.n_components))
        log_beta = np.zeros((n_samples, self.n_components))

        # Forward
        log_emission = self._log_emission_prob(X)
        log_alpha[0] = np.log(self.startprob_ + 1e-16) + log_emission[0]
        for t in range(1, n_samples):
            for j in range(self.n_components):
                log_alpha[t, j] = log_emission[t, j] + self._logsumexp(
                    log_alpha[t - 1] + np.log(self.transmat_[:, j] + 1e-16)
                )

        log_likelihood = self._logsumexp(log_alpha[-1])

        # Backward
        log_beta[-1] = 0.0
        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_components):
                log_beta[t, i] = self._logsumexp(
                    log_beta[t + 1] + log_emission[t + 1] + np.log(self.transmat_[i] + 1e-16)
                )

        return log_alpha, log_beta, log_likelihood

    def _compute_log_xi(
        self, X: np.ndarray, log_alpha: np.ndarray, log_beta: np.ndarray, log_likelihood: float
    ) -> np.ndarray:
        """计算log xi (transition posterior)。"""
        n_samples = len(X)
        log_xi = np.zeros((n_samples - 1, self.n_components, self.n_components))
        log_emission = self._log_emission_prob(X)

        for t in range(n_samples - 1):
            for i in range(self.n_components):
                for j in range(self.n_components):
                    log_xi[t, i, j] = (
                        log_alpha[t, i]
                        + np.log(self.transmat_[i, j] + 1e-16)
                        + log_emission[t + 1, j]
                        + log_beta[t + 1, j]
                        - log_likelihood
                    )
        return log_xi

    def _log_emission_prob(self, X: np.ndarray) -> np.ndarray:
        """计算对数发射概率。"""
        n_samples = len(X)
        log_prob = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            diff = X - self.means_[k]
            var = self.covars_[k]
            log_prob[:, k] = -0.5 * np.sum((diff**2) / var + np.log(2 * np.pi * var), axis=1)
        return log_prob

    @staticmethod
    def _logsumexp(a: np.ndarray) -> float:
        """稳定的log-sum-exp。"""
        a_max = a.max()
        return a_max + np.log(np.sum(np.exp(a - a_max)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回每个时间点的regime概率分布。"""
        log_alpha, log_beta, log_likelihood = self._forward_backward(X)
        log_gamma = log_alpha + log_beta - log_likelihood
        return np.exp(log_gamma)

    def decode(self, X: np.ndarray) -> np.ndarray:
        """Viterbi解码。"""
        n_samples = len(X)
        log_emission = self._log_emission_prob(X)
        log_delta = np.zeros((n_samples, self.n_components))
        psi = np.zeros((n_samples, self.n_components), dtype=int)

        log_delta[0] = np.log(self.startprob_ + 1e-16) + log_emission[0]

        for t in range(1, n_samples):
            for j in range(self.n_components):
                vals = log_delta[t - 1] + np.log(self.transmat_[:, j] + 1e-16)
                psi[t, j] = np.argmax(vals)
                log_delta[t, j] = vals[psi[t, j]] + log_emission[t, j]

        # Backtracking
        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(log_delta[-1])
        for t in range(n_samples - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states


class HMMRegimeDetector:
    """
    Layer 2: HMM数据驱动regime检测。

    使用高斯隐马尔可夫模型自动识别市场regime。
    hmmlearn可用时使用其GaussianHMM，否则使用numpy简化实现。

    Parameters
    ----------
    n_regimes : int, default 3
        隐状态（regime）数量。
    random_state : int, default 42
        随机种子。
    """

    def __init__(self, n_regimes: int = 3, random_state: int = 42) -> None:
        self.n_regimes = n_regimes
        self.random_state = random_state
        self._model: Optional[Any] = None
        self._is_fitted: bool = False

    def fit(
        self, returns: pd.Series, volatilities: pd.Series
    ) -> Dict[str, Any]:
        r"""
        用EM算法训练HMM。

        Parameters
        ----------
        returns : pd.Series
            收益时间序列。
        volatilities : pd.Series
            波动率时间序列。

        Returns
        -------
        dict
            {
                'regime_probs': DataFrame (每个时间点的regime概率),
                'transition_matrix': ndarray (状态转移矩阵),
                'means': ndarray (每个regime的均值向量),
                'covars': ndarray (每个regime的协方差),
            }
        """
        # 对齐并构造特征矩阵
        aligned_ret, aligned_vol = returns.align(volatilities, join="inner")
        X = np.column_stack([aligned_ret.values, aligned_vol.values])
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)  # 标准化

        if _HMMLEARN_AVAILABLE:
            self._model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="diag",
                random_state=self.random_state,
                n_iter=100,
                tol=1e-3,
            )
            self._model.fit(X)
            transmat = self._model.transmat_
            means = self._model.means_
            covars = self._model.covars_
            regime_probs = pd.DataFrame(
                self._model.predict_proba(X),
                index=aligned_ret.index,
                columns=[f"regime_{i}" for i in range(self.n_regimes)],
            )
        else:
            np_fallback = _NumpyGaussianHMM(
                n_components=self.n_regimes, n_iter=100
            )
            np_fallback.fit(X)
            self._model = np_fallback
            transmat = np_fallback.transmat_
            means = np_fallback.means_
            covars = np_fallback.covars_
            regime_probs = pd.DataFrame(
                np_fallback.predict_proba(X),
                index=aligned_ret.index,
                columns=[f"regime_{i}" for i in range(self.n_regimes)],
            )

        self._is_fitted = True

        return {
            "regime_probs": regime_probs,
            "transition_matrix": transmat,
            "means": means,
            "covars": covars,
        }

    def predict(
        self, returns: pd.Series, volatilities: pd.Series
    ) -> pd.DataFrame:
        """
        返回每个时间点的regime概率分布。

        Parameters
        ----------
        returns : pd.Series
            收益时间序列。
        volatilities : pd.Series
            波动率时间序列。

        Returns
        -------
        pd.DataFrame
            列为 regime_0, regime_1, ..., index为日期。
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        aligned_ret, aligned_vol = returns.align(volatilities, join="inner")
        X = np.column_stack([aligned_ret.values, aligned_vol.values])
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        if _HMMLEARN_AVAILABLE:
            probs = self._model.predict_proba(X)
        else:
            probs = self._model.predict_proba(X)

        return pd.DataFrame(
            probs,
            index=aligned_ret.index,
            columns=[f"regime_{i}" for i in range(self.n_regimes)],
        )

    def decode(
        self, returns: pd.Series, volatilities: pd.Series
    ) -> np.ndarray:
        """
        Viterbi解码，返回最可能的regime序列。

        Parameters
        ----------
        returns : pd.Series
            收益时间序列。
        volatilities : pd.Series
            波动率时间序列。

        Returns
        -------
        np.ndarray
            最可能的隐状态序列（0, 1, ..., n_regimes-1）。
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        aligned_ret, aligned_vol = returns.align(volatilities, join="inner")
        X = np.column_stack([aligned_ret.values, aligned_vol.values])
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        if _HMMLEARN_AVAILABLE:
            states = self._model.predict(X)
        else:
            states = self._model.decode(X)

        return states


# ---------------------------------------------------------------------------
# Layer 3: Regime Conditional IC
# ---------------------------------------------------------------------------

def _rank_ic_numpy(factor_vals: np.ndarray, forward_returns: np.ndarray) -> float:
    """纯numpy计算Rank IC。"""
    mask = ~(np.isnan(factor_vals) | np.isnan(forward_returns))
    if mask.sum() < 3:
        return np.nan
    f, r = factor_vals[mask], forward_returns[mask]
    ranks_f = np.argsort(np.argsort(f, kind="mergesort")) + 1
    ranks_r = np.argsort(np.argsort(r, kind="mergesort")) + 1
    return float(np.corrcoef(ranks_f, ranks_r)[0, 1])


def _welch_ttest(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Welch's t-test（不假设等方差）。

    Returns
    -------
    Tuple[float, float]
        (t_statistic, p_value)
    """
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan, np.nan

    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)

    if vx == 0 and vy == 0:
        return 0.0, 1.0

    se = np.sqrt(vx / nx + vy / ny)
    t_stat = (mx - my) / se

    # Welch-Satterthwaite自由度
    df = (vx / nx + vy / ny) ** 2 / (
        (vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1)
    )

    if _SCIPY_AVAILABLE:
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    else:
        # 简化：使用正态近似
        p_value = 2 * min(
            1.0,
            np.exp(-0.5 * t_stat**2)
            * np.sqrt(2 / np.pi)
            / (abs(t_stat) + 1e-8),
        )

    return float(t_stat), float(p_value)


class RegimeConditionalIC:
    r"""
    Layer 3: Regime Conditional Factor Performance — 核心验证层。

    计算每个regime下每个因子的Rank IC，检验不同regime下IC差异的显著性，
    并验证当前regime权重配置是否有数据支撑。
    """

    def calculate_conditional_ic(
        self,
        factor_df: pd.DataFrame,
        factor_cols: List[str],
        return_col: str = "forward_return",
        regime_col: str = "regime",
    ) -> pd.DataFrame:
        """
        计算每个regime下每个因子的Rank IC。

        Parameters
        ----------
        factor_df : pd.DataFrame
            包含因子值、前向收益和regime标签的DataFrame。
        factor_cols : List[str]
            因子列名列表。
        return_col : str, default 'forward_return'
            前向收益列名。
        regime_col : str, default 'regime'
            regime标签列名。

        Returns
        -------
        pd.DataFrame
            列为: factor | regime | rank_ic | ic_std | n_obs
        """
        records = []

        for factor_col in factor_cols:
            if factor_col not in factor_df.columns:
                continue

            for regime, group in factor_df.groupby(regime_col):
                ic_list = []
                # Determine grouping key for date
                if "date" in group.columns:
                    date_groups = group.groupby("date")
                else:
                    date_groups = group.groupby(group.index)
                for date, sub in date_groups:
                    f = sub[factor_col].values
                    r = sub[return_col].values
                    ic = _rank_ic_numpy(f, r)
                    if not np.isnan(ic):
                        ic_list.append(ic)

                if ic_list:
                    records.append(
                        {
                            "factor": factor_col,
                            "regime": regime,
                            "rank_ic": round(np.mean(ic_list), 4),
                            "ic_std": round(np.std(ic_list, ddof=1), 4),
                            "n_obs": len(ic_list),
                        }
                    )

        return pd.DataFrame(records)

    def test_significance(
        self, conditional_ic: pd.DataFrame
    ) -> pd.DataFrame:
        """
        用Welch's t-test检验不同regime下IC的差异是否显著。

        Parameters
        ----------
        conditional_ic : pd.DataFrame
            calculate_conditional_ic的输出。

        Returns
        -------
        pd.DataFrame
            列为: factor | regime_pair | ic_diff | t_stat | p_value | significant
        """
        records = []
        factors = conditional_ic["factor"].unique()

        for factor in factors:
            sub = conditional_ic[conditional_ic["factor"] == factor]
            regimes = sub["regime"].unique()

            for i in range(len(regimes)):
                for j in range(i + 1, len(regimes)):
                    r1, r2 = regimes[i], regimes[j]
                    ic1 = sub[sub["regime"] == r1]["rank_ic"].values
                    ic2 = sub[sub["regime"] == r2]["rank_ic"].values

                    if len(ic1) >= 3 and len(ic2) >= 3:
                        t_stat, p_value = _welch_ttest(ic1, ic2)
                        records.append(
                            {
                                "factor": factor,
                                "regime_pair": f"{r1}_vs_{r2}",
                                "ic_diff": round(float(np.mean(ic1) - np.mean(ic2)), 4),
                                "t_stat": round(t_stat, 4),
                                "p_value": round(p_value, 4),
                                "significant": p_value < 0.05,
                            }
                        )

        return pd.DataFrame(records)

    def validate_regime_weights(
        self,
        conditional_ic: pd.DataFrame,
        current_weights: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        验证当前regime权重是否有数据支撑。

        Parameters
        ----------
        conditional_ic : pd.DataFrame
            条件IC数据。
        current_weights : dict
            {regime: {factor: weight}} 当前权重配置。

        Returns
        -------
        dict
            {regime: {factor: {'current_weight': w, 'ic_supported': bool, 'recommended_weight': w}}}
        """
        result: Dict[str, Any] = {}

        for regime, factor_weights in current_weights.items():
            result[regime] = {}
            for factor, weight in factor_weights.items():
                ic_data = conditional_ic[
                    (conditional_ic["regime"] == regime)
                    & (conditional_ic["factor"] == factor)
                ]

                if ic_data.empty:
                    result[regime][factor] = {
                        "current_weight": weight,
                        "ic_supported": False,
                        "recommended_weight": 0.0,
                        "note": "No IC data available",
                    }
                else:
                    rank_ic = ic_data["rank_ic"].values[0]
                    ic_supported = abs(rank_ic) >= 0.03
                    recommended = weight if ic_supported else 0.0
                    result[regime][factor] = {
                        "current_weight": weight,
                        "ic_supported": ic_supported,
                        "recommended_weight": recommended,
                        "rank_ic": rank_ic,
                    }

        return result

    def get_regime_factor_ranking(
        self, conditional_ic: pd.DataFrame, regime: str
    ) -> pd.DataFrame:
        """
        返回指定regime下按IC排序的因子排名。

        Parameters
        ----------
        conditional_ic : pd.DataFrame
            条件IC数据。
        regime : str
            Regime名称。

        Returns
        -------
        pd.DataFrame
            按|Rank IC|降序排列的因子排名。
        """
        sub = conditional_ic[conditional_ic["regime"] == regime].copy()
        sub["abs_ic"] = sub["rank_ic"].abs()
        sub = sub.sort_values("abs_ic", ascending=False)
        return sub[["factor", "rank_ic", "ic_std", "n_obs"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def estimate_half_life(ic_decay_series: pd.Series) -> Optional[float]:
    """
    从IC衰减序列估计半衰期。

    Parameters
    ----------
    ic_decay_series : pd.Series
        index=lag, values=Rank IC。

    Returns
    -------
    Optional[float]
        IC半衰期（月）。若无法估计返回None。
    """
    clean = ic_decay_series.dropna()
    if len(clean) < 3:
        return None

    lags = clean.index.values.astype(float)
    ics = clean.values.astype(float)

    # 用指数衰减模型拟合: IC(lag) = IC(0) * exp(-lambda * lag)
    # log(IC) = log(IC_0) - lambda * lag
    log_ics = np.log(np.abs(ics) + 1e-8)
    A = np.column_stack([np.ones(len(lags)), -lags])
    try:
        beta = np.linalg.lstsq(A, log_ics, rcond=None)[0]
        lam = beta[1]
        if lam > 0:
            return float(np.log(2) / lam)
    except Exception:
        pass

    # Fallback: 找到IC衰减到50%的lag
    ic0 = ics[0]
    target = abs(ic0) * 0.5
    for lag, ic in clean.items():
        if abs(ic) <= target:
            return float(lag)

    return None


# ---------------------------------------------------------------------------
# 用法示例
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    n = 120  # 10年月度数据
    dates = pd.date_range("2014-01-01", periods=n, freq="MS")
    returns = pd.Series(np.random.randn(n) * 0.05, index=dates, name="returns")
    volatilities = pd.Series(
        np.abs(np.random.randn(n)) * 0.15 + 0.1, index=dates, name="volatility"
    )

    # Layer 1: Rule-based
    print("=" * 60)
    print("Layer 1: Rule-based Regime Detection")
    print("=" * 60)
    rb = RuleBasedRegime()
    regimes = rb.detect_series(volatilities, returns)
    print(regimes.value_counts())
    print()

    # Layer 2: HMM
    print("=" * 60)
    print("Layer 2: HMM Regime Detection")
    print("=" * 60)
    hmm = HMMRegimeDetector(n_regimes=3)
    result = hmm.fit(returns, volatilities)
    print("Transition Matrix:")
    print(result["transition_matrix"])
    print("\nRegime Means:")
    print(result["means"])
    print()

    # Layer 3: Regime Conditional IC
    print("=" * 60)
    print("Layer 3: Regime Conditional IC")
    print("=" * 60)
    # 构造带regime标签的因子数据
    factor_data = []
    stocks = [f"STK_{i:03d}" for i in range(50)]
    for d in dates:
        regime = regimes.loc[d]
        for s in stocks:
            factor_data.append(
                {
                    "date": d,
                    "stock": s,
                    "regime": regime,
                    "pe_ratio": np.random.randn(),
                    "momentum_1m": np.random.randn(),
                    "forward_return": np.random.randn() * 0.05,
                }
            )
    factor_df = pd.DataFrame(factor_data)

    rc = RegimeConditionalIC()
    cond_ic = rc.calculate_conditional_ic(
        factor_df, ["pe_ratio", "momentum_1m"]
    )
    print(cond_ic.to_string(index=False))
