"""
Benchmark Models — Champion Challenger Framework

模型层级：
1. Gatekeeper Baseline: Equal Weighted Linear, IC Weighted Linear
2. Linear ML: Ridge / Elastic Net
3. Nonlinear: Random Forest, XGBoost, LightGBM
4. Advanced: LambdaMART (Learning to Rank)

依赖降级策略：
- sklearn: 未安装时用numpy实现coordinate descent简化版Elastic Net
- xgboost: 未安装时跳过
- lightgbm: 未安装时fallback到XGBoost或跳过
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "BaseScorer",
    "EqualWeightedScorer",
    "ICWeightedScorer",
    "ElasticNetScorer",
    "RandomForestScorer",
    "XGBoostScorer",
    "LightGBMScorer",
    "ModelComparison",
]

# ---------------------------------------------------------------------------
# 依赖探测与降级
# ---------------------------------------------------------------------------

_SKLEARN_AVAILABLE = False
_XGBOOST_AVAILABLE = False
_LIGHTGBM_AVAILABLE = False

try:
    from sklearn.linear_model import ElasticNet as _SklearnElasticNet
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    _SKLEARN_AVAILABLE = True
except ImportError:
    warnings.warn("sklearn not available; ElasticNet will use numpy fallback.")

try:
    import xgboost as xgb

    _XGBOOST_AVAILABLE = True
except ImportError:
    warnings.warn("xgboost not available; XGBoostScorer will be disabled.")

try:
    import lightgbm as lgb

    _LIGHTGBM_AVAILABLE = True
except ImportError:
    warnings.warn("lightgbm not available; LightGBMScorer will be disabled.")


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class BaseScorer(ABC):
    """
    所有评分模型的抽象基类。

    子类必须实现 fit, predict, get_model_name 三个方法。
    """

    @abstractmethod
    def fit(
        self,
        factor_df: pd.DataFrame,
        factor_cols: List[str],
        target_col: str,
    ) -> None:
        """训练模型。"""

    @abstractmethod
    def predict(self, factor_df: pd.DataFrame) -> np.ndarray:
        """预测评分。"""

    @abstractmethod
    def get_model_name(self) -> str:
        """返回模型名称。"""

    @abstractmethod
    def get_model_type(self) -> str:
        """返回模型类型 (linear / nonlinear / ltr)."""

    @abstractmethod
    def get_model_role(self) -> str:
        """返回模型角色 (gatekeeper / challenger / advanced)."""


# ---------------------------------------------------------------------------
# Layer 1: Gatekeeper Baseline
# ---------------------------------------------------------------------------

class EqualWeightedScorer(BaseScorer):
    r"""
    等权线性baseline — 最透明、零过拟合风险。

    Score = mean(factor_values * direction)

    direction 由因子定义确定：+1 表示正向因子，-1 表示反向因子。
    若未指定方向，默认使用+1。

    Parameters
    ----------
    directions : Optional[Dict[str, float]]
        每个因子的方向字典，如 {'pe_ratio': -1, 'momentum_1m': +1}。
    """

    def __init__(self, directions: Optional[Dict[str, float]] = None) -> None:
        self.directions = directions or {}
        self._factor_cols: List[str] = []

    def fit(
        self,
        factor_df: pd.DataFrame,
        factor_cols: List[str],
        target_col: str,
    ) -> None:
        """等权模型无需训练，仅记录因子列。"""
        self._factor_cols = factor_cols

    def predict(self, factor_df: pd.DataFrame) -> np.ndarray:
        """
        等权合成因子值。

        Returns
        -------
        np.ndarray
            每个样本的综合评分。
        """
        if not self._factor_cols:
            raise RuntimeError("Model not fitted yet")
        vals = factor_df[self._factor_cols].fillna(0).values
        dirs = np.array([self.directions.get(c, 1.0) for c in self._factor_cols])
        weighted = vals * dirs
        return weighted.mean(axis=1)

    def get_model_name(self) -> str:
        return "EqualWeighted"

    def get_model_type(self) -> str:
        return "linear"

    def get_model_role(self) -> str:
        return "gatekeeper"


class ICWeightedScorer(BaseScorer):
    r"""
    IC加权线性模型 — 用过去12-24个月的滚动Rank IC作为权重。

    weight_i = mean(Rank_IC_i) / std(Rank_IC_i)
    score    = sum(weight_i * factor_i)

    Parameters
    ----------
    lookback_months : int, default 12
        计算滚动IC的回顾月数。
    """

    def __init__(self, lookback_months: int = 12) -> None:
        self.lookback_months = lookback_months
        self._weights: Optional[np.ndarray] = None
        self._factor_cols: List[str] = []

    def fit(
        self,
        factor_df: pd.DataFrame,
        factor_cols: List[str],
        target_col: str,
    ) -> None:
        """
        用历史IC计算权重。

        对每个因子计算滚动Rank IC的均值/标准差作为权重。
        """
        self._factor_cols = factor_cols
        weights = []

        for col in factor_cols:
            ic_list = []
            for date, group in factor_df.groupby(level=0 if factor_df.index.nlevels > 1 else factor_df.columns[0] if factor_df.columns[0] in factor_df else 'date'):
                pass

        # 更robust的实现: 要求factor_df含date列
        if "date" not in factor_df.columns:
            raise ValueError("factor_df must contain 'date' column for ICWeightedScorer")

        for col in factor_cols:
            ic_list = []
            for date, group in factor_df.groupby("date"):
                f = group[col]
                t = group[target_col]
                mask = f.notna() & t.notna()
                if mask.sum() < 5:
                    continue
                # Spearman rank IC (simplified, via corr on ranks)
                ranked_f = f[mask].rank()
                ranked_t = t[mask].rank()
                ic = ranked_f.corr(ranked_t, method="pearson")
                if not np.isnan(ic):
                    ic_list.append(ic)

            if len(ic_list) >= 3:
                mean_ic = np.mean(ic_list[-self.lookback_months:])
                std_ic = np.std(ic_list[-self.lookback_months:], ddof=1)
                weight = mean_ic / (std_ic + 1e-8)
            else:
                weight = 0.0
            weights.append(weight)

        self._weights = np.array(weights)

    def predict(self, factor_df: pd.DataFrame) -> np.ndarray:
        """IC加权预测。"""
        if self._weights is None:
            raise RuntimeError("Model not fitted yet")
        vals = factor_df[self._factor_cols].fillna(0).values
        return vals @ self._weights

    def get_model_name(self) -> str:
        return "ICWeighted"

    def get_model_type(self) -> str:
        return "linear"

    def get_model_role(self) -> str:
        return "gatekeeper"


# ---------------------------------------------------------------------------
# Layer 2: Linear ML — Elastic Net (with numpy fallback)
# ---------------------------------------------------------------------------

class _NumpyElasticNet:
    """
    纯numpy实现的Elastic Net（coordinate descent简化版）。

    当sklearn不可用时作为fallback。支持L1+L2正则化。
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NumpyElasticNet":
        """Coordinate descent训练。"""
        n_samples, n_features = X.shape
        # 标准化
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0, ddof=1) + 1e-8
        X_scaled = (X - self._x_mean) / self._x_std
        y_mean = y.mean()
        y_centered = y - y_mean

        beta = np.zeros(n_features)
        lambda_total = self.alpha * n_samples
        lambda_l1 = self.l1_ratio * lambda_total
        lambda_l2 = (1 - self.l1_ratio) * lambda_total

        for iteration in range(self.max_iter):
            beta_old = beta.copy()
            for j in range(n_features):
                residual = y_centered - X_scaled @ beta + X_scaled[:, j] * beta[j]
                rho = X_scaled[:, j] @ residual
                z = np.sum(X_scaled[:, j] ** 2) + lambda_l2

                if rho < -lambda_l1:
                    beta[j] = (rho + lambda_l1) / z
                elif rho > lambda_l1:
                    beta[j] = (rho - lambda_l1) / z
                else:
                    beta[j] = 0.0

            if np.linalg.norm(beta - beta_old, ord=1) < self.tol:
                break

        self.coef_ = beta / self._x_std
        self.intercept_ = y_mean - (self.coef_ * self._x_mean).sum()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测。"""
        if self.coef_ is None:
            raise RuntimeError("Model not fitted yet")
        return X @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R^2 score。"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-12)


class ElasticNetScorer(BaseScorer):
    """
    Elastic Net — 稳健线性ML，防共线性过拟合。

    优先使用sklearn实现，不可用时自动降级为numpy coordinate descent。

    Parameters
    ----------
    alpha : float, default 1.0
        正则化强度。
    l1_ratio : float, default 0.5
        L1/L2混合比例，0=Ridge, 1=Lasso。
    """

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self._model: Optional[Any] = None
        self._factor_cols: List[str] = []

    def fit(
        self,
        factor_df: pd.DataFrame,
        factor_cols: List[str],
        target_col: str,
    ) -> None:
        """训练Elastic Net模型。"""
        self._factor_cols = factor_cols
        X = factor_df[factor_cols].fillna(0).values
        y = factor_df[target_col].values

        if _SKLEARN_AVAILABLE:
            self._model = _SklearnElasticNet(
                alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=2000
            )
            self._model.fit(X, y)
        else:
            self._model = _NumpyElasticNet(
                alpha=self.alpha, l1_ratio=self.l1_ratio
            )
            self._model.fit(X, y)

    def predict(self, factor_df: pd.DataFrame) -> np.ndarray:
        """预测评分。"""
        if self._model is None:
            raise RuntimeError("Model not fitted yet")
        X = factor_df[self._factor_cols].fillna(0).values
        return self._model.predict(X)

    def get_model_name(self) -> str:
        return f"ElasticNet(alpha={self.alpha}, l1_ratio={self.l1_ratio})"

    def get_model_type(self) -> str:
        return "linear"

    def get_model_role(self) -> str:
        return "challenger"


# ---------------------------------------------------------------------------
# Layer 3: Nonlinear Models
# ---------------------------------------------------------------------------

class RandomForestScorer(BaseScorer):
    """
    Random Forest — 非线性对照组。

    sklearn可用时启用，否则抛出ImportError。

    Parameters
    ----------
    n_estimators : int, default 100
        树的数量。
    max_depth : int, default 6
        最大深度（控制过拟合）。
    random_state : int, default 42
        随机种子。
    """

    def __init__(
        self, n_estimators: int = 100, max_depth: int = 6, random_state: int = 42
    ) -> None:
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for RandomForestScorer. "
                "Install it: pip install scikit-learn"
            )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._model: Optional[Any] = None
        self._factor_cols: List[str] = []

    def fit(
        self,
        factor_df: pd.DataFrame,
        factor_cols: List[str],
        target_col: str,
    ) -> None:
        """训练Random Forest模型。"""
        self._factor_cols = factor_cols
        X = factor_df[factor_cols].fillna(0).values
        y = factor_df[target_col].values
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X, y)

    def predict(self, factor_df: pd.DataFrame) -> np.ndarray:
        """预测评分。"""
        if self._model is None:
            raise RuntimeError("Model not fitted yet")
        X = factor_df[self._factor_cols].fillna(0).values
        return self._model.predict(X)

    def get_model_name(self) -> str:
        return f"RandomForest(depth={self.max_depth})"

    def get_model_type(self) -> str:
        return "nonlinear"

    def get_model_role(self) -> str:
        return "challenger"


class XGBoostScorer(BaseScorer):
    """
    XGBoost — 梯度提升决策树。

    xgboost可用时启用，否则抛出ImportError。

    Parameters
    ----------
    n_estimators : int, default 100
         boosting轮数。
    max_depth : int, default 4
        树的最大深度。
    learning_rate : float, default 0.05
        学习率。
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
    ) -> None:
        if not _XGBOOST_AVAILABLE:
            raise ImportError(
                "xgboost is required for XGBoostScorer. "
                "Install it: pip install xgboost"
            )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._model: Optional[Any] = None
        self._factor_cols: List[str] = []

    def fit(
        self,
        factor_df: pd.DataFrame,
        factor_cols: List[str],
        target_col: str,
    ) -> None:
        """训练XGBoost模型。"""
        self._factor_cols = factor_cols
        X = factor_df[factor_cols].fillna(0).values
        y = factor_df[target_col].values
        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X, y)

    def predict(self, factor_df: pd.DataFrame) -> np.ndarray:
        """预测评分。"""
        if self._model is None:
            raise RuntimeError("Model not fitted yet")
        X = factor_df[self._factor_cols].fillna(0).values
        return self._model.predict(X)

    def get_model_name(self) -> str:
        return f"XGBoost(depth={self.max_depth})"

    def get_model_type(self) -> str:
        return "nonlinear"

    def get_model_role(self) -> str:
        return "challenger"


class LightGBMScorer(BaseScorer):
    """
    LightGBM — 未来扩展方向，支持native ranking objective。

    lightgbm可用时启用；不可用时尝试fallback到XGBoost；
    两者都不可用时抛出ImportError。

    Parameters
    ----------
    n_estimators : int, default 100
        boosting轮数。
    max_depth : int, default 4
        树的最大深度。
    learning_rate : float, default 0.05
        学习率。
    use_ranking : bool, default False
        是否使用ranking objective（需要额外group信息）。
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        use_ranking: bool = False,
    ) -> None:
        self._use_lgb = _LIGHTGBM_AVAILABLE
        self._use_xgb = _XGBOOST_AVAILABLE

        if not self._use_lgb and not self._use_xgb:
            raise ImportError(
                "LightGBMScorer requires lightgbm or xgboost. "
                "Install: pip install lightgbm xgboost"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.use_ranking = use_ranking
        self._model: Optional[Any] = None
        self._factor_cols: List[str] = []

    def fit(
        self,
        factor_df: pd.DataFrame,
        factor_cols: List[str],
        target_col: str,
    ) -> None:
        """训练LightGBM（或fallback到XGBoost）模型。"""
        self._factor_cols = factor_cols
        X = factor_df[factor_cols].fillna(0).values
        y = factor_df[target_col].values

        if self._use_lgb:
            objective = "lambdarank" if self.use_ranking else "regression"
            self._model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                objective=objective if not self.use_ranking else "regression",
            )
            self._model.fit(X, y)
        elif self._use_xgb:
            warnings.warn("LightGBM not available, falling back to XGBoost.")
            self._model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
            self._model.fit(X, y)

    def predict(self, factor_df: pd.DataFrame) -> np.ndarray:
        """预测评分。"""
        if self._model is None:
            raise RuntimeError("Model not fitted yet")
        X = factor_df[self._factor_cols].fillna(0).values
        return self._model.predict(X)

    def get_model_name(self) -> str:
        backend = "LightGBM" if self._use_lgb else "XGBoostFallback"
        return f"{backend}(depth={self.max_depth})"

    def get_model_type(self) -> str:
        return "nonlinear"

    def get_model_role(self) -> str:
        return "challenger"


# ---------------------------------------------------------------------------
# 模型对比框架
# ---------------------------------------------------------------------------

def _rank_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算排名相关系数。"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 3:
        return np.nan
    ranks_t = np.argsort(np.argsort(y_true[mask], kind="mergesort")) + 1
    ranks_p = np.argsort(np.argsort(y_pred[mask], kind="mergesort")) + 1
    return float(np.corrcoef(ranks_t, ranks_p)[0, 1])


def _calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """计算一组评估指标。"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]

    # R^2
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    # Rank correlation
    rank_corr = _rank_correlation(yt, yp)

    # Hit rate (direction accuracy)
    hit_rate = float(np.mean(np.sign(yt) == np.sign(yp)))

    # Sharpe (assume monthly data, annualize)
    excess = yp - yt.mean()
    sharpe = float(excess.mean() / (excess.std(ddof=1) + 1e-12) * np.sqrt(12))

    # Max drawdown
    cumulative = np.cumsum(yp)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_dd = float(drawdown.min())

    # Information Ratio
    tracking_error = (yp - yt).std(ddof=1)
    ir = float((yp.mean() - yt.mean()) / (tracking_error + 1e-12) * np.sqrt(12))

    return {
        "r2": round(r2, 4),
        "rank_corr": round(rank_corr, 4),
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 4),
        "hit_rate": round(hit_rate, 4),
        "ir": round(ir, 4),
    }


class ModelComparison:
    """
    运行所有模型并生成对比报告。

    Parameters
    ----------
    None

    Attributes
    ----------
    results : pd.DataFrame
        对比结果表格。
    """

    def __init__(self) -> None:
        self.results: Optional[pd.DataFrame] = None

    def run_comparison(
        self,
        factor_df: pd.DataFrame,
        factor_cols: List[str],
        target_col: str,
        models: Optional[List[BaseScorer]] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        r"""
        运行所有benchmark模型，返回对比表格。

        Parameters
        ----------
        factor_df : pd.DataFrame
            训练数据。
        factor_cols : List[str]
            因子列名。
        target_col : str
            目标列名。
        models : Optional[List[BaseScorer]]
            自定义模型列表。None时使用默认集合。
        test_df : Optional[pd.DataFrame]
            测试数据。None时使用factor_df作为测试集（in-sample）。

        Returns
        -------
        pd.DataFrame
            columns: model | type | role | r2 | rank_corr | sharpe | max_dd | hit_rate | ir
        """
        if models is None:
            models = self._default_models()

        eval_df = test_df if test_df is not None else factor_df
        records = []

        for model in models:
            try:
                model.fit(factor_df, factor_cols, target_col)
                y_pred = model.predict(eval_df)
                y_true = eval_df[target_col].values
                metrics = _calculate_metrics(y_true, y_pred)
                records.append(
                    {
                        "model": model.get_model_name(),
                        "type": model.get_model_type(),
                        "role": model.get_model_role(),
                        **metrics,
                    }
                )
            except Exception as e:
                records.append(
                    {
                        "model": model.get_model_name(),
                        "type": model.get_model_type(),
                        "role": model.get_model_role(),
                        "r2": np.nan,
                        "rank_corr": np.nan,
                        "sharpe": np.nan,
                        "max_dd": np.nan,
                        "hit_rate": np.nan,
                        "ir": np.nan,
                        "error": str(e),
                    }
                )

        self.results = pd.DataFrame(records)
        return self.results

    def get_best_model(self, metric: str = "ir") -> str:
        """
        返回指定metric下表现最好的模型。

        Parameters
        ----------
        metric : str, default 'ir'
            排序指标名。

        Returns
        -------
        str
            最优模型名称。
        """
        if self.results is None:
            raise RuntimeError("Must run run_comparison() first")
        best_idx = self.results[metric].idxmax()
        return str(self.results.loc[best_idx, "model"])

    def get_beaten_baselines(self, candidate_model: BaseScorer) -> List[str]:
        """
        检查候选模型是否跑赢所有gatekeeper baseline。

        Parameters
        ----------
        candidate_model : BaseScorer
            待检验的候选模型。

        Returns
        -------
        List[str]
            被候选模型击败的baseline名称列表。若未击败任何baseline返回空列表。
        """
        if self.results is None:
            raise RuntimeError("Must run run_comparison() first")

        gatekeepers = self.results[self.results["role"] == "gatekeeper"]
        if gatekeepers.empty:
            return []

        candidate_row = self.results[
            self.results["model"] == candidate_model.get_model_name()
        ]
        if candidate_row.empty:
            return []

        beaten = []
        candidate_ir = candidate_row["ir"].values[0]
        for _, gk in gatekeepers.iterrows():
            if candidate_ir > gk["ir"]:
                beaten.append(str(gk["model"]))

        return beaten

    @staticmethod
    def _default_models() -> List[BaseScorer]:
        """返回默认模型列表（仅包含无需外部依赖的模型）。"""
        models: List[BaseScorer] = [
            EqualWeightedScorer(),
            ICWeightedScorer(),
            ElasticNetScorer(),
        ]

        if _SKLEARN_AVAILABLE:
            models.append(RandomForestScorer())
        if _XGBOOST_AVAILABLE:
            models.append(XGBoostScorer())
        if _LIGHTGBM_AVAILABLE:
            models.append(LightGBMScorer())

        return models


# ---------------------------------------------------------------------------
# 用法示例
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    n = 5000
    df = pd.DataFrame(
        {
            "pe_ratio": np.random.randn(n),
            "pb_ratio": np.random.randn(n),
            "momentum_1m": np.random.randn(n),
            "volatility_20d": np.random.randn(n),
            "market_cap": np.random.randn(n),
            "forward_return": np.random.randn(n) * 0.05 + 0.001,
        }
    )
    factor_cols = ["pe_ratio", "pb_ratio", "momentum_1m", "volatility_20d", "market_cap"]

    # 运行对比
    comparison = ModelComparison()
    results = comparison.run_comparison(df, factor_cols, "forward_return")
    print("=" * 70)
    print("Benchmark Model Comparison — Champion Challenger Framework")
    print("=" * 70)
    print(results.to_string(index=False))
    print(f"\nBest Model (by IR): {comparison.get_best_model('ir')}")
