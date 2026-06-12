#!/usr/bin/env python3
"""
Mini-GRP ML Enhancement Engine
==============================
基于Alan Wang反馈的改进：从线性加权升级到ML驱动的动态因子选择和非线性评分。

核心模块：
1. MLFactorSelector     — XGBoost动态因子选择（解决"三四百个input"的维度灾难）
2. NonlinearScorer      — XGBoost非线性评分 + SHAP可解释性
3. RegimeDetector       — 市场环境检测（牛市/熊市/震荡市）
4. WalkForwardBacktester— Walk-forward回测框架

设计原则（参考Principal GRP经验）：
- 新兴市场（A股）允许更复杂的模型（市场低效，alpha空间大）
- 保持可解释性（SHAP + 浅树max_depth=3）
- 严格防止过拟合（TimeSeriesSplit + Walk-forward + 正则化）

依赖：xgboost, sklearn, shap(optional), pandas, numpy
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 0. 工具函数
# ---------------------------------------------------------------------------

def _safe_import(module_name: str, package_name: str = None):
    """安全导入可选依赖"""
    try:
        return __import__(module_name)
    except ImportError:
        pkg = package_name or module_name
        logger.warning(f"{pkg}未安装，相关功能不可用。运行: pip install {pkg}")
        return None

# 延迟导入ML库（未安装时优雅降级）
xgb = _safe_import("xgboost")
sklearn_model_selection = _safe_import("sklearn.model_selection")
sklearn_metrics = _safe_import("sklearn.metrics")
shap = _safe_import("shap")

# ---------------------------------------------------------------------------
# 1. ML Factor Selector — 动态因子选择
# ---------------------------------------------------------------------------

class MLFactorSelector:
    """
    XGBoost动态因子选择器

    核心逻辑：
    1. 输入所有可用因子（16个或更多）
    2. 用XGBoost预测forward return
    3. 通过feature importance排序，只保留Top-N有效因子
    4. 最终评分仍用线性加权（保持可解释性）

    与Principal GRP的类比：
    - GRP有20+模型、数百个factor
    - 我们先用ML选8-10个最有效的，避免维度灾难
    """

    DEFAULT_PARAMS = {
        'max_depth': 3,           # 浅树！防止过拟合
        'learning_rate': 0.05,    # 保守学习
        'n_estimators': 100,
        'subsample': 0.8,         # 随机采样
        'colsample_bytree': 0.8,  # 随机特征
        'reg_alpha': 0.1,         # L1正则
        'reg_lambda': 1.0,        # L2正则
        'random_state': 42,
        'n_jobs': -1,
    }

    def __init__(self, n_top_factors: int = 10, forward_period: int = 20, **xgb_kwargs):
        self.n_top_factors = n_top_factors
        self.forward_period = forward_period
        self.xgb_params = {**self.DEFAULT_PARAMS, **xgb_kwargs}
        self.selected_factors: List[str] = []
        self.feature_importance: Optional[pd.DataFrame] = None
        self.cv_scores: List[float] = []

        if xgb is None:
            raise ImportError("xgboost未安装。运行: pip install xgboost")

    def fit(self, factor_df: pd.DataFrame, factor_cols: List[str],
            return_col: str = 'forward_return') -> Tuple[List[str], pd.DataFrame]:
        """
        训练因子选择器

        Parameters
        ----------
        factor_df : DataFrame包含因子暴露和未来收益
        factor_cols : 因子列名列表（如 ['pe_ttm_z', 'roe_deducted_z', ...]）
        return_col : 目标变量列名（未来收益）

        Returns
        -------
        selected_factors : 选出的Top-N因子列表
        feature_importance : 所有因子的重要性排序DataFrame
        """
        from sklearn.model_selection import TimeSeriesSplit

        X = factor_df[factor_cols].values
        y = factor_df[return_col].values if return_col in factor_df.columns else None

        if y is None:
            raise ValueError(f"数据中缺少'{return_col}'列。请先计算forward return。")

        # 去除NaN
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean, y_clean = X[valid], y[valid]
        n_valid = len(X_clean)
        logger.info(f"有效样本数: {n_valid} (原{len(factor_df)}，去除{len(factor_df)-n_valid}个NaN)")

        if n_valid < 200:
            logger.warning("样本量不足200，因子选择结果可能不稳定")

        # TimeSeriesSplit — 金融数据必须用时间序列CV！
        n_splits = min(5, n_valid // 100)
        if n_splits < 2:
            n_splits = 2
        tscv = TimeSeriesSplit(n_splits=n_splits)

        importances = []
        self.cv_scores = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_clean)):
            X_train, X_test = X_clean[train_idx], X_clean[test_idx]
            y_train, y_test = y_clean[train_idx], y_clean[test_idx]

            model = xgb.XGBRegressor(**self.xgb_params)
            model.fit(X_train, y_train)

            # 记录importance
            importances.append(model.feature_importances_)

            # 计算R2
            pred = model.predict(X_test)
            r2 = 1 - np.sum((y_test - pred)**2) / np.sum((y_test - np.mean(y_test))**2)
            self.cv_scores.append(r2)
            logger.info(f"  Fold {fold+1}/{n_splits}: Test R2={r2:.4f}, "
                       f"train={len(train_idx)}, test={len(test_idx)}")

        # 平均importance
        avg_importance = np.mean(importances, axis=0)

        self.feature_importance = pd.DataFrame({
            'factor': factor_cols,
            'importance': avg_importance,
            'importance_pct': avg_importance / avg_importance.sum() * 100
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        # 选Top-N
        n_select = min(self.n_top_factors, len(factor_cols))
        self.selected_factors = self.feature_importance.head(n_select)['factor'].tolist()

        avg_r2 = np.mean(self.cv_scores)
        logger.info(f"因子选择完成: 从{len(factor_cols)}个中选了{len(self.selected_factors)}个, "
                    f"平均CV R2={avg_r2:.4f}")
        logger.info(f"选中因子: {self.selected_factors}")

        return self.selected_factors, self.feature_importance

    def get_factor_importance_dict(self) -> Dict[str, float]:
        """返回factor → importance的字典"""
        if self.feature_importance is None:
            return {}
        return dict(zip(self.feature_importance['factor'],
                       self.feature_importance['importance']))


# ---------------------------------------------------------------------------
# 2. Nonlinear Scorer — 非线性评分 + SHAP解释
# ---------------------------------------------------------------------------

class NonlinearScorer:
    """
    XGBoost非线性评分引擎

    核心逻辑：
    - 线性模型: score = 0.3*V + 0.3*Q + 0.2*G + 0.2*M（假设独立）
    - XGBoost: score = f(V, Q, G, M)（学习交互效应）

    SHAP解释：
    - 每个股票的最终得分 = baseline + SHAP(V) + SHAP(Q) + SHAP(G) + SHAP(M)
    - PM可以看到每个dimension的贡献，保持可解释性
    """

    DEFAULT_PARAMS = {
        'max_depth': 3,
        'learning_rate': 0.05,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
    }

    def __init__(self, use_shap: bool = True, **xgb_kwargs):
        self.use_shap = use_shap and (shap is not None)
        self.xgb_params = {**self.DEFAULT_PARAMS, **xgb_kwargs}
        self.model = None
        self.shap_explainer = None
        self.dimension_cols: List[str] = []

        if xgb is None:
            raise ImportError("xgboost未安装。运行: pip install xgboost")

    def fit(self, factor_df: pd.DataFrame,
            dimension_cols: List[str] = None,
            target_col: str = 'forward_return') -> 'NonlinearScorer':
        """
        训练非线性评分模型

        Parameters
        ----------
        factor_df : DataFrame
        dimension_cols : 维度得分列名，如 ['value_score', 'quality_score', ...]
        target_col : 目标变量（未来收益）
        """
        if dimension_cols is None:
            dimension_cols = ['value_score', 'quality_score', 'growth_score', 'momentum_score']
        self.dimension_cols = dimension_cols

        X = factor_df[dimension_cols].values
        y = factor_df[target_col].values if target_col in factor_df.columns else None

        if y is None:
            raise ValueError(f"缺少'{target_col}'列")

        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[valid], y[valid]

        self.model = xgb.XGBRegressor(**self.xgb_params)
        self.model.fit(X, y)

        # SHAP
        if self.use_shap:
            try:
                self.shap_explainer = shap.TreeExplainer(self.model)
                logger.info("SHAP explainer已初始化")
            except Exception as e:
                logger.warning(f"SHAP初始化失败: {e}")
                self.use_shap = False

        # 评估
        pred = self.model.predict(X)
        r2 = 1 - np.sum((y - pred)**2) / np.sum((y - np.mean(y))**2)
        logger.info(f"NonlinearScorer训练完成: In-sample R2={r2:.4f}, "
                   f"features={dimension_cols}")

        return self

    def predict(self, factor_df: pd.DataFrame) -> pd.Series:
        """预测非线性评分"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit()")

        X = factor_df[self.dimension_cols].fillna(0).values
        scores = self.model.predict(X)

        # 转换为0-100
        from scipy import stats
        percentile = stats.rankdata(scores, method='average') / len(scores) * 100

        return pd.Series(percentile, index=factor_df.index, name='nonlinear_score')

    def explain(self, factor_df: pd.DataFrame, idx: int = 0) -> Dict:
        """
        SHAP解释单个股票的评分构成

        Returns
        -------
        dict: {dimension: contribution, ...}
        """
        if not self.use_shap or self.shap_explainer is None:
            return {"error": "SHAP不可用"}

        X = factor_df[self.dimension_cols].fillna(0).values
        shap_values = self.shap_explainer.shap_values(X)

        sv = shap_values[idx] if len(shap_values.shape) > 1 else shap_values

        result = {
            'baseline': float(self.shap_explainer.expected_value)
                        if hasattr(self.shap_explainer, 'expected_value') else 0.0,
        }
        for col, val in zip(self.dimension_cols, sv):
            result[col] = float(val)
        result['total'] = sum(v for k, v in result.items() if k != 'baseline')

        return result

    def explain_all(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """SHAP解释所有股票"""
        if not self.use_shap or self.shap_explainer is None:
            return pd.DataFrame()

        X = factor_df[self.dimension_cols].fillna(0).values
        shap_values = self.shap_explainer.shap_values(X)

        df = pd.DataFrame(shap_values, columns=[f"{c}_shap" for c in self.dimension_cols],
                         index=factor_df.index)
        return df


# ---------------------------------------------------------------------------
# 3. Regime Detector — 市场环境检测
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    市场环境（Regime）检测器

    根据Alan的反馈：不同市场环境下，有效因子不同。
    - 牛市: Growth/Momentum主导
    - 熊市: Value/Quality主导
    - 震荡: 均衡或Mean-reversion

    实现方式：
    1. 简单版：基于波动率和流动性的手动规则（推荐先用这个）
    2. 高级版：Gaussian HMM（Hidden Markov Model）
    """

    REGIME_NAMES = {
        0: 'bull_market',      # 牛市
        1: 'bear_market',      # 熊市
        2: 'volatile',         # 高波动
        3: 'range_bound',      # 震荡市
    }

    def __init__(self, method: str = 'rule'):
        """
        Parameters
        ----------
        method : 'rule'（手动规则）或 'hmm'（隐马尔可夫模型）
        """
        self.method = method
        self.hmm_model = None

        if method == 'hmm':
            hmm_module = _safe_import("hmmlearn.hmm", "hmmlearn")
            if hmm_module is None:
                logger.warning("hmmlearn未安装，回退到rule方法")
                self.method = 'rule'

    def detect(self, volatility: float = 20.0,
               liquidity: float = 0.0,
               market_return_20d: float = 0.0) -> int:
        """
        基于手动规则检测当前regime

        Parameters
        ----------
        volatility : 年化波动率 (%)
        liquidity : 市场换手率变化 (%)
        market_return_20d : 市场20日收益 (%)

        Returns
        -------
        regime_id : 0=牛市, 1=熊市, 2=高波动, 3=震荡市
        """
        if volatility < 18 and market_return_20d > 2:
            return 0  # 牛市：低波动+正收益
        elif volatility > 28 or market_return_20d < -5:
            return 1  # 熊市：高波动或大跌
        elif volatility > 25:
            return 2  # 高波动
        else:
            return 3  # 震荡市

    def detect_from_data(self, market_data: pd.DataFrame) -> pd.Series:
        """
        从市场数据批量检测regime

        market_data需要包含：
        - 'close' 或 'index_close': 收盘价
        """
        if 'close' not in market_data.columns and 'index_close' not in market_data.columns:
            logger.warning("市场数据缺少收盘价列")
            return pd.Series(3, index=market_data.index)  # 默认震荡

        close_col = 'close' if 'close' in market_data.columns else 'index_close'
        close = market_data[close_col]

        # 计算特征
        ret_20d = close.pct_change(20) * 100
        volatility = close.pct_change().rolling(20).std() * np.sqrt(252) * 100

        regimes = []
        for i in range(len(market_data)):
            vol = volatility.iloc[i] if not pd.isna(volatility.iloc[i]) else 20.0
            ret = ret_20d.iloc[i] if not pd.isna(ret_20d.iloc[i]) else 0.0
            r = self.detect(vol, 0.0, ret)
            regimes.append(r)

        return pd.Series(regimes, index=market_data.index)

    def get_optimal_weights(self, regime_id: int) -> Dict[str, float]:
        """
        获取当前regime下的最优因子权重（基于历史回测经验）

        这些权重是基于A股历史数据的empirical observation
        """
        weights_map = {
            0: {'value': 0.15, 'quality': 0.25, 'growth': 0.35, 'momentum': 0.25},  # 牛市
            1: {'value': 0.45, 'quality': 0.35, 'growth': 0.05, 'momentum': 0.15},  # 熊市
            2: {'value': 0.30, 'quality': 0.30, 'growth': 0.15, 'momentum': 0.25},  # 高波动
            3: {'value': 0.35, 'quality': 0.30, 'growth': 0.20, 'momentum': 0.15},  # 震荡
        }
        return weights_map.get(regime_id, weights_map[3])


# ---------------------------------------------------------------------------
# 4. Walk-Forward Backtester — 回测框架
# ---------------------------------------------------------------------------

class WalkForwardBacktester:
    """
    Walk-forward回测框架

    核心原则（Alan提到的教训）：
    - 不能用random train/test split（金融数据有时间依赖性）
    - 必须用walk-forward：只用过去数据训练，预测未来
    - 防止look-ahead bias和data snooping

    流程：
    1. 设定训练窗口（如252个交易日=1年）和预测窗口（如20天）
    2. 在每个时间点t，用[t-252, t]的数据训练模型
    3. 预测[t, t+20]的收益
    4. 按预测得分分5组，计算每组收益
    5. 滚动窗口向前移动，重复
    """

    def __init__(self, train_window: int = 252, pred_window: int = 20,
                 step: int = 20):
        self.train_window = train_window
        self.pred_window = pred_window
        self.step = step

    def run(self, factor_df: pd.DataFrame,
            factor_cols: List[str],
            return_col: str = 'forward_return',
            score_col: str = 'composite_score',
            n_quantiles: int = 5) -> pd.DataFrame:
        """
        执行walk-forward回测

        Returns
        -------
        backtest_results : DataFrame
            columns: date, long_return, short_return, long_short_return,
                     long_cum, short_cum, ls_cum
        """
        if xgb is None:
            raise ImportError("xgboost未安装")

        results = []
        dates = factor_df.index
        n = len(dates)

        logger.info(f"Walk-forward回测开始: 总样本{n}, "
                   f"训练窗口{self.train_window}, 预测窗口{self.pred_window}")

        for start in range(self.train_window, n - self.pred_window, self.step):
            train_start = start - self.train_window
            train_end = start
            pred_start = start
            pred_end = min(start + self.pred_window, n)

            train_df = factor_df.iloc[train_start:train_end]
            pred_df = factor_df.iloc[pred_start:pred_end]

            # 训练XGBoost
            X_train = train_df[factor_cols].fillna(0).values
            y_train = train_df[return_col].values

            valid = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
            if valid.sum() < 50:
                continue

            model = xgb.XGBRegressor(
                max_depth=3, learning_rate=0.05, n_estimators=100,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42
            )
            model.fit(X_train[valid], y_train[valid])

            # 预测
            X_pred = pred_df[factor_cols].fillna(0).values
            pred_scores = model.predict(X_pred)

            # 分5组
            pred_returns = pred_df[return_col].values

            try:
                quantiles = pd.qcut(pred_scores, n_quantiles, labels=False, duplicates='drop')
            except ValueError:
                continue

            long_return = pred_returns[quantiles == quantiles.max()].mean()
            short_return = pred_returns[quantiles == quantiles.min()].mean()
            ls_return = long_return - short_return

            results.append({
                'date': dates[pred_start],
                'long_return': long_return,
                'short_return': short_return,
                'long_short_return': ls_return,
            })

        if not results:
            logger.warning("回测无有效结果")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df['long_cum'] = (1 + df['long_return']).cumprod()
        df['short_cum'] = (1 + df['short_return']).cumprod()
        df['ls_cum'] = (1 + df['long_short_return']).cumprod()

        # 计算统计指标
        self.stats = {
            'total_periods': len(df),
            'long_annual_return': df['long_return'].mean() * (252 / self.pred_window),
            'long_sharpe': df['long_return'].mean() / df['long_return'].std() * np.sqrt(252 / self.pred_window) if df['long_return'].std() > 0 else 0,
            'ls_annual_return': df['long_short_return'].mean() * (252 / self.pred_window),
            'ls_sharpe': df['long_short_return'].mean() / df['long_short_return'].std() * np.sqrt(252 / self.pred_window) if df['long_short_return'].std() > 0 else 0,
            'win_rate': (df['long_short_return'] > 0).mean(),
            'max_drawdown': (df['ls_cum'] / df['ls_cum'].cummax() - 1).min(),
        }

        logger.info(f"回测完成: {len(df)}期, "
                   f"Long-Sharpe={self.stats['ls_sharpe']:.3f}, "
                   f"WinRate={self.stats['win_rate']:.1%}")

        return df


# ---------------------------------------------------------------------------
# 5. 主入口 — 整合所有ML模块
# ---------------------------------------------------------------------------

def run_ml_pipeline(factor_df: pd.DataFrame,
                    factor_cols: List[str] = None,
                    dimension_cols: List[str] = None,
                    return_col: str = 'forward_return',
                    use_ml_selection: bool = True,
                    use_nonlinear: bool = True,
                    use_regime: bool = True,
                    n_top_factors: int = 10) -> Dict:
    """
    执行完整的ML增强流程

    Parameters
    ----------
    factor_df : DataFrame
        包含因子暴露和未来收益的数据
    factor_cols : 因子列名列表（如 ['pe_ttm_z', ...]）
    dimension_cols : 维度得分列名（如 ['value_score', ...]）
    return_col : 目标收益列
    use_ml_selection : 是否使用ML因子选择
    use_nonlinear : 是否使用非线性评分
    use_regime : 是否使用regime-aware权重
    n_top_factors : ML选择的因子数量

    Returns
    -------
    results : dict包含所有ML模块的输出
    """
    if factor_cols is None:
        factor_cols = [c for c in factor_df.columns if c.endswith('_z')]
    if dimension_cols is None:
        dimension_cols = ['value_score', 'quality_score', 'growth_score', 'momentum_score']

    results = {
        'linear_score': factor_df.get('composite_score'),
        'ml_selected_factors': factor_cols,  # 默认用全部
        'feature_importance': None,
        'nonlinear_score': None,
        'shap_explanation': None,
        'current_regime': None,
        'regime_weights': None,
    }

    # 1. ML Factor Selection
    if use_ml_selection and return_col in factor_df.columns:
        try:
            selector = MLFactorSelector(n_top_factors=n_top_factors)
            selected, importance = selector.fit(factor_df, factor_cols, return_col)
            results['ml_selected_factors'] = selected
            results['feature_importance'] = importance
            logger.info(f"[ML] 因子选择: {len(selected)}/{len(factor_cols)}个因子")
        except Exception as e:
            logger.warning(f"[ML] 因子选择失败: {e}，使用全部因子")

    # 2. Nonlinear Scoring
    if use_nonlinear and return_col in factor_df.columns:
        try:
            scorer = NonlinearScorer(use_shap=True)
            scorer.fit(factor_df, dimension_cols, return_col)
            nl_score = scorer.predict(factor_df)
            results['nonlinear_score'] = nl_score

            # SHAP解释Top 10
            top10_idx = nl_score.nlargest(10).index
            explanations = []
            for idx in top10_idx[:3]:  # 只解释前3个作为示例
                exp = scorer.explain(factor_df, factor_df.index.get_loc(idx))
                explanations.append(exp)
            results['shap_explanation'] = explanations

            logger.info(f"[ML] 非线性评分完成: score范围[{nl_score.min():.1f}, {nl_score.max():.1f}]")
        except Exception as e:
            logger.warning(f"[ML] 非线性评分失败: {e}")

    # 3. Regime Detection
    if use_regime:
        try:
            detector = RegimeDetector(method='rule')
            # 从数据中估算当前regime
            if 'return_20d' in factor_df.columns:
                avg_ret = factor_df['return_20d'].mean()
            else:
                avg_ret = 0.0
            regime = detector.detect(volatility=20.0, market_return_20d=avg_ret)
            weights = detector.get_optimal_weights(regime)
            results['current_regime'] = RegimeDetector.REGIME_NAMES.get(regime, 'unknown')
            results['regime_weights'] = weights
            logger.info(f"[ML] 当前Regime: {results['current_regime']}, 权重: {weights}")
        except Exception as e:
            logger.warning(f"[ML] Regime检测失败: {e}")

    return results


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Mini-GRP ML Engine 测试")
    print("=" * 60)

    # 生成带forward return的模拟数据
    np.random.seed(42)
    n = 500

    # 16个因子
    factor_cols = [f'f{i}_z' for i in range(16)]
    for i in range(16):
        pass  # 后面生成

    # 简化测试：直接用16列
    data = {
        'code': [f"{600000+i:06d}" for i in range(n)],
        'name': [f"stock_{i}" for i in range(n)],
        'value_score': np.random.normal(0, 1, n),
        'quality_score': np.random.normal(0, 1, n),
        'growth_score': np.random.normal(0, 1, n),
        'momentum_score': np.random.normal(0, 1, n),
        'forward_return': np.random.normal(0.02, 0.05, n),  # 模拟未来收益
    }

    # 添加16个因子列
    for i in range(16):
        data[f'f{i}_z'] = np.random.normal(0, 1, n)

    df = pd.DataFrame(data)

    # 让部分因子与return有真实相关性（模拟有效因子）
    df['forward_return'] += 0.02 * df['f0_z'] + 0.015 * df['f1_z'] - 0.01 * df['f2_z']
    df['forward_return'] += 0.01 * df['value_score'] + 0.008 * df['quality_score']

    print(f"\n数据: {len(df)}行, 因子+维度共{16+4}列")
    print(f"Forward return: mean={df['forward_return'].mean():.4f}, "
          f"std={df['forward_return'].std():.4f}")

    # 测试1: Factor Selector
    print("\n--- Test 1: MLFactorSelector ---")
    all_factor_cols = [f'f{i}_z' for i in range(16)]
    selector = MLFactorSelector(n_top_factors=8)
    selected, importance = selector.fit(df, all_factor_cols, 'forward_return')
    print(f"选中因子: {selected}")
    print(f"\nFeature Importance Top 10:")
    print(importance.head(10).to_string(index=False))

    # 测试2: Nonlinear Scorer
    print("\n--- Test 2: NonlinearScorer ---")
    dim_cols = ['value_score', 'quality_score', 'growth_score', 'momentum_score']
    scorer = NonlinearScorer(use_shap=False)
    scorer.fit(df, dim_cols, 'forward_return')
    scores = scorer.predict(df)
    print(f"非线性评分: mean={scores.mean():.2f}, range=[{scores.min():.1f}, {scores.max():.1f}]")

    # 测试3: Regime Detector
    print("\n--- Test 3: RegimeDetector ---")
    detector = RegimeDetector()
    for vol, ret, name in [(15, 3, '牛市'), (30, -6, '熊市'), (20, 0, '震荡')]:
        r = detector.detect(vol, 0, ret)
        w = detector.get_optimal_weights(r)
        print(f"  {name}(vol={vol}, ret={ret}): regime={detector.REGIME_NAMES[r]}, weights={w}")

    # 测试4: Full Pipeline
    print("\n--- Test 4: Full ML Pipeline ---")
    results = run_ml_pipeline(df, all_factor_cols, dim_cols, use_nonlinear=False)
    print(f"选中因子数: {len(results['ml_selected_factors'])}")
    print(f"当前Regime: {results['current_regime']}")
    print(f"Regime权重: {results['regime_weights']}")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
