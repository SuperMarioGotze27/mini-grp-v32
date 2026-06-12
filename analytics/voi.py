"""Value of Information — Data Acquisition Decision Framework
对应课程：Lecture 2 (Party Problem, Value of a Forecaster)

核心逻辑：
    - 当前模型 performance → expected profit
    - 获取新数据后 improved performance → improved expected profit
    - Value of Information = improved_profit - current_profit
    - 如果 VoI > cost_of_data → 值得获取

课程背景：
    Lecture 2 介绍了 Value of Information 概念，通过 Party Problem
    说明：获取额外信息（如天气预报）的决策价值 = 有信息时的期望效用
    - 无信息时的期望效用。在量化投资中，等价于：
    - 获取新数据后的预期 IR 提升带来的管理费收入增长。

应用场景：
    1. 评估是否购买新的数据源（Wind, Bloomberg 等）
    2. 评估是否扩大股票 universe
    3. 评估是否雇佣新的分析师团队
    4. 数据预算分配决策
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

__all__ = [
    "value_of_information",
    "data_source_voi_analysis",
    "expansion_decision_voi",
    "ValueOfInformationFramework",
]


def value_of_information(
    current_ir: float,
    improved_ir: float,
    aum: float = 100_000_000,
    management_fee: float = 0.015,
    data_cost: float = 0.0,
) -> Dict:
    """
    计算获取新数据的信息价值 (VoI)

    基本假设：
        - 管理费率收入 ∝ IR × AUM × 管理费率
        - IR 提升 → 管理规模增长 → 收入增长
        - 收入增量 = VoI

    计算公式：
        profit = base_profit + scaling_factor × IR × AUM × management_fee
        current_profit = scaling_factor × current_ir × AUM × management_fee
        improved_profit = scaling_factor × improved_ir × AUM × management_fee
        value_of_information = improved_profit - current_profit
        roi = value_of_information / data_cost

    Parameters
    ----------
    current_ir : float
        当前 Information Ratio。
        IR = (portfolio_return - benchmark_return) / tracking_error
    improved_ir : float
        获取新数据后预期的 IR。
    aum : float, default 100_000_000
        资产管理规模（美元）。
    management_fee : float, default 0.015
        年度管理费率（如 0.015 = 1.5%）。
    data_cost : float, default 0.0
        获取数据的年度成本（美元）。用于计算 ROI。

    Returns
    -------
    Dict
        包含以下键：
        - current_profit (float): 当前年度管理费收入
        - improved_profit (float): 改善后年度管理费收入
        - value_of_information (float): VoI = 年收入增量
        - data_cost (float): 数据成本
        - roi (float): VoI / data_cost，如果 data_cost = 0 则为 inf
        - recommendation (str): "Acquire" if VoI > data_cost else "Not Worth It"

    Notes
    -----
    - IR 提升假设是可持续的（即长期有效）
    - 管理费率收入模型简化：实际收入还受客户留存、市场波动等影响
    - 对于 data_cost = 0 的内部数据（如新的数据处理方式），roi = inf

    Examples
    --------
    >>> result = value_of_information(current_ir=0.8, improved_ir=1.0, aum=1e8,
    ...                               management_fee=0.015, data_cost=5000)
    >>> print(f"VoI = ${result['value_of_information']:,.0f}")
    >>> print(f"ROI = {result['roi']:.1f}x")
    >>> print(f"Recommendation: {result['recommendation']}")
    """
    # 收入计算：假设收入与 IR 线性相关（IR 越高，越能吸引资金）
    # 使用 IR 作为 alpha 能力的代理，收入 = IR * AUM * fee_rate
    current_profit = current_ir * aum * management_fee
    improved_profit = improved_ir * aum * management_fee

    voi = improved_profit - current_profit

    if data_cost > 0:
        roi = voi / data_cost
        recommendation = "Acquire" if voi > data_cost else "Not Worth It"
    else:
        roi = float("inf")
        recommendation = "Acquire (zero cost)"

    return {
        "current_profit": round(current_profit, 2),
        "improved_profit": round(improved_profit, 2),
        "value_of_information": round(voi, 2),
        "data_cost": round(data_cost, 2),
        "roi": round(roi, 2) if roi != float("inf") else float("inf"),
        "recommendation": recommendation,
    }


def data_source_voi_analysis(
    aum: float = 100_000_000,
    current_ir: float = 0.8,
    management_fee: float = 0.015,
) -> pd.DataFrame:
    """
    对各种数据源做 VoI 分析

    数据源清单（基于市场常见定价和效果估计）：
    - Wind (A股/港股): 成本 $5,000/年, 预期 IR 提升 0.05
    - Bloomberg (全球): 成本 $24,000/年, 预期 IR 提升 0.08
    - iFinD (A股): 成本 $3,000/年, 预期 IR 提升 0.03
    - NLP 舆情数据: 成本 $12,000/年, 预期 IR 提升 0.06
    - 卫星/另类数据: 成本 $50,000/年, 预期 IR 提升 0.04
    - Refinitiv (全球): 成本 $18,000/年, 预期 IR 提升 0.07

    Parameters
    ----------
    aum : float, default 100_000_000
        资产管理规模（美元）。
    current_ir : float, default 0.8
        当前 Information Ratio。
    management_fee : float, default 0.015
        年度管理费率。

    Returns
    -------
    pd.DataFrame
        columns: data_source | cost | ir_improvement | improved_ir |
                 voi | roi | recommendation
        按 roi 降序排列。

    Examples
    --------
    >>> df = data_source_voi_analysis(aum=200_000_000, current_ir=0.9)
    >>> print(df[['data_source', 'cost', 'voi', 'roi', 'recommendation']].to_string(index=False))
    """
    data_sources = [
        {
            "data_source": "Wind (A股/港股)",
            "cost": 5000,
            "ir_improvement": 0.05,
        },
        {
            "data_source": "Bloomberg (全球)",
            "cost": 24000,
            "ir_improvement": 0.08,
        },
        {
            "data_source": "iFinD (A股)",
            "cost": 3000,
            "ir_improvement": 0.03,
        },
        {
            "data_source": "NLP舆情数据",
            "cost": 12000,
            "ir_improvement": 0.06,
        },
        {
            "data_source": "卫星/另类数据",
            "cost": 50000,
            "ir_improvement": 0.04,
        },
        {
            "data_source": "Refinitiv (全球)",
            "cost": 18000,
            "ir_improvement": 0.07,
        },
    ]

    rows = []
    for ds in data_sources:
        improved_ir = current_ir + ds["ir_improvement"]
        result = value_of_information(
            current_ir=current_ir,
            improved_ir=improved_ir,
            aum=aum,
            management_fee=management_fee,
            data_cost=ds["cost"],
        )
        rows.append(
            {
                "data_source": ds["data_source"],
                "cost": ds["cost"],
                "ir_improvement": ds["ir_improvement"],
                "improved_ir": improved_ir,
                "voi": result["value_of_information"],
                "roi": result["roi"],
                "recommendation": result["recommendation"],
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("roi", ascending=False)

    return df


def expansion_decision_voi(
    current_universe_size: int,
    target_universe_size: int,
    current_ir: float,
    aum: float = 100_000_000,
    management_fee: float = 0.015,
    expansion_cost: float = 0.0,
) -> Dict:
    """
    计算扩大 universe（如从 150 只 → 500 只）的 VoI

    假设：universe 扩大后，diversification benefit 增加，IR 提升。
    参考公式：IR_new ≈ IR_current × sqrt(target_size / current_size)

    这个公式来自主动管理基本定律（Fundamental Law of Active Management）：
        IR ≈ IC × sqrt(BR)
    其中 BR = breadth = universe size。
    因此 IR ∝ sqrt(universe_size)。

    Parameters
    ----------
    current_universe_size : int
        当前股票池大小。
    target_universe_size : int
        目标股票池大小（必须 > current_universe_size）。
    current_ir : float
        当前 Information Ratio。
    aum : float, default 100_000_000
        资产管理规模（美元）。
    management_fee : float, default 0.015
        年度管理费率。
    expansion_cost : float, default 0.0
        扩大 universe 的年度成本（如额外研究人员、数据成本等）。

    Returns
    -------
    Dict
        包含以下键：
        - current_universe_size (int): 当前 universe 大小
        - target_universe_size (int): 目标 universe 大小
        - current_ir (float): 当前 IR
        - expected_ir (float): 预期新 IR
        - ir_improvement (float): IR 提升量
        - voi (float): 信息价值
        - expansion_cost (float): 扩大成本
        - roi (float): VoI / expansion_cost
        - recommendation (str): 决策建议

    Notes
    -----
    - IR ∝ sqrt(N) 的假设只在信息比率独立时成立
    - 实际中，universe 扩大可能导致 IC 下降（信息质量稀释）
    - 此模型未考虑 IC 稀释效应，是乐观估计

    Examples
    --------
    >>> result = expansion_decision_voi(
    ...     current_universe_size=150, target_universe_size=500,
    ...     current_ir=0.8, aum=1e8, expansion_cost=100000
    ... )
    >>> print(f"Expected IR: {result['expected_ir']:.4f}")
    >>> print(f"VoI: ${result['voi']:,.0f}")
    >>> print(f"Recommendation: {result['recommendation']}")
    """
    if target_universe_size <= current_universe_size:
        raise ValueError(
            f"target_universe_size ({target_universe_size}) must be greater than "
            f"current_universe_size ({current_universe_size})"
        )

    if current_universe_size <= 0:
        raise ValueError("current_universe_size must be positive")

    # IR_new ≈ IR_current × sqrt(target_size / current_size)
    size_ratio = target_universe_size / current_universe_size
    expected_ir = current_ir * np.sqrt(size_ratio)

    # VoI 计算
    voi_result = value_of_information(
        current_ir=current_ir,
        improved_ir=expected_ir,
        aum=aum,
        management_fee=management_fee,
        data_cost=expansion_cost,
    )

    roi = voi_result["roi"]
    recommendation = (
        "Expand Universe"
        if voi_result["recommendation"].startswith("Acquire")
        else "Maintain Current Universe"
    )

    return {
        "current_universe_size": current_universe_size,
        "target_universe_size": target_universe_size,
        "current_ir": current_ir,
        "expected_ir": round(expected_ir, 4),
        "ir_improvement": round(expected_ir - current_ir, 4),
        "voi": voi_result["value_of_information"],
        "expansion_cost": expansion_cost,
        "roi": roi,
        "recommendation": recommendation,
    }


class ValueOfInformationFramework:
    """
    Value of Information 统一接口

    封装 VoI 分析的全流程，支持：
    - 评估单个数据源的获取价值
    - 生成数据获取优先级排序
    - 扩展 universe 的决策分析

    Parameters
    ----------
    current_metrics : Dict
        当前策略指标，需包含：
        - 'ir' (float): 当前 Information Ratio
        - 'aum' (float): 资产管理规模（美元）
        - 'management_fee' (float): 管理费率

    Attributes
    ----------
    current_ir : float
    aum : float
    management_fee : float

    Examples
    --------
    >>> metrics = {'ir': 0.8, 'aum': 100_000_000, 'management_fee': 0.015}
    >>> voi = ValueOfInformationFramework(metrics)
    >>> result = voi.evaluate_data_acquisition('wind')
    >>> priority = voi.get_investment_priority()
    """

    def __init__(self, current_metrics: Dict):
        required_keys = {"ir", "aum", "management_fee"}
        missing = required_keys - set(current_metrics.keys())
        if missing:
            raise ValueError(f"Missing required keys in current_metrics: {missing}")

        self.current_ir = float(current_metrics["ir"])
        self.aum = float(current_metrics["aum"])
        self.management_fee = float(current_metrics["management_fee"])

        # 内置数据源配置
        self._data_source_configs: Dict[str, Dict] = {
            "wind": {
                "name": "Wind (A股/港股)",
                "cost": 5000,
                "ir_improvement": 0.05,
            },
            "bloomberg": {
                "name": "Bloomberg (全球)",
                "cost": 24000,
                "ir_improvement": 0.08,
            },
            "ifind": {
                "name": "iFinD (A股)",
                "cost": 3000,
                "ir_improvement": 0.03,
            },
            "nlp": {
                "name": "NLP舆情数据",
                "cost": 12000,
                "ir_improvement": 0.06,
            },
            "satellite": {
                "name": "卫星/另类数据",
                "cost": 50000,
                "ir_improvement": 0.04,
            },
            "refinitiv": {
                "name": "Refinitiv (全球)",
                "cost": 18000,
                "ir_improvement": 0.07,
            },
        }

    def evaluate_data_acquisition(self, data_source: str) -> Dict:
        """
        评估获取某个数据源的价值

        Parameters
        ----------
        data_source : str
            数据源标识符：
            - 'wind': Wind (A股/港股)
            - 'bloomberg': Bloomberg (全球)
            - 'ifind': iFinD (A股)
            - 'nlp': NLP 舆情数据
            - 'satellite': 卫星/另类数据
            - 'refinitiv': Refinitiv (全球)

        Returns
        -------
        Dict
            value_of_information 函数的返回值，额外包含 'data_source_name'。

        Examples
        --------
        >>> result = voi.evaluate_data_acquisition('wind')
        >>> print(result['recommendation'])
        'Acquire'
        """
        config = self._data_source_configs.get(data_source)
        if config is None:
            available = ", ".join(self._data_source_configs.keys())
            raise ValueError(
                f"Unknown data_source: '{data_source}'. Available: {available}"
            )

        improved_ir = self.current_ir + config["ir_improvement"]
        result = value_of_information(
            current_ir=self.current_ir,
            improved_ir=improved_ir,
            aum=self.aum,
            management_fee=self.management_fee,
            data_cost=config["cost"],
        )
        result["data_source_name"] = config["name"]

        return result

    def get_investment_priority(self) -> pd.DataFrame:
        """
        按 ROI 排序的数据获取优先级列表

        Returns
        -------
        pd.DataFrame
            columns: data_source | data_source_name | cost | ir_improvement |
                     voi | roi | recommendation
            按 roi 降序排列。

        Examples
        --------
        >>> priority = voi.get_investment_priority()
        >>> print(priority[['data_source_name', 'roi', 'recommendation']].head(3))
        """
        rows = []
        for key, config in self._data_source_configs.items():
            result = self.evaluate_data_acquisition(key)
            rows.append(
                {
                    "data_source": key,
                    "data_source_name": config["name"],
                    "cost": config["cost"],
                    "ir_improvement": config["ir_improvement"],
                    "voi": result["value_of_information"],
                    "roi": result["roi"],
                    "recommendation": result["recommendation"],
                }
            )

        df = pd.DataFrame(rows)
        df = df.sort_values("roi", ascending=False).reset_index(drop=True)

        return df


# ───────────────────────────────────────────────────────────────────────────────
# 用法示例（可独立运行）
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Demo: Value of Information")
    print("=" * 60)

    # 1) 基本的 VoI 计算
    print("\n[1] Basic VoI Calculation:")
    result = value_of_information(
        current_ir=0.8,
        improved_ir=1.0,
        aum=100_000_000,
        management_fee=0.015,
        data_cost=5000,
    )
    for k, v in result.items():
        if isinstance(v, float):
            print(f"    {k}: {v:,.2f}")
        else:
            print(f"    {k}: {v}")

    # 2) 数据源 VoI 分析
    print(f"\n[2] Data Source VoI Analysis (AUM = $100M, current IR = 0.8):")
    ds_analysis = data_source_voi_analysis(
        aum=100_000_000, current_ir=0.8, management_fee=0.015
    )
    cols = ["data_source", "cost", "ir_improvement", "voi", "roi", "recommendation"]
    print(ds_analysis[cols].to_string(index=False))

    # 3) Universe 扩展决策
    print(f"\n[3] Universe Expansion Decision:")
    expansion = expansion_decision_voi(
        current_universe_size=150,
        target_universe_size=500,
        current_ir=0.8,
        aum=100_000_000,
        management_fee=0.015,
        expansion_cost=100_000,
    )
    for k, v in expansion.items():
        if isinstance(v, float):
            print(f"    {k}: {v:,.2f}" if abs(v) > 10 else f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    # 4) ValueOfInformationFramework 类
    print(f"\n[4] ValueOfInformationFramework:")
    metrics = {"ir": 0.8, "aum": 100_000_000, "management_fee": 0.015}
    framework = ValueOfInformationFramework(metrics)

    # 评估 Wind 数据
    wind_eval = framework.evaluate_data_acquisition("wind")
    print(f"    Wind evaluation: {wind_eval['recommendation']} (ROI: {wind_eval['roi']:.1f}x)")

    # 优先级排序
    priority = framework.get_investment_priority()
    print(f"\n    Investment Priority (top 3):")
    for _, row in priority.head(3).iterrows():
        print(
            f"      {row['data_source_name']}: "
            f"ROI = {row['roi']:.1f}x, "
            f"VoI = ${row['voi']:,.0f}"
        )

    # 5) 边界情况测试
    print(f"\n[5] Edge case — zero cost data:")
    free_result = value_of_information(
        current_ir=0.8, improved_ir=0.85, aum=1e8, management_fee=0.015, data_cost=0
    )
    print(f"    Zero-cost data: {free_result['recommendation']}, ROI = {free_result['roi']}")

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
