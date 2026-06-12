"""
Model Governance & Champion Challenger Framework

机构级模型管理制度：
- Champion: 当前production模型
- Challenger: 候选新模型
- Gatekeeper Baseline: 必须跑赢的门槛
- Shadow: 仅在shadow环境运行
- Degradation Triggers: 模型降级触发条件

所有代码仅依赖标准库 + numpy/pandas，无需外部依赖。
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "ModelRole",
    "DegradationTrigger",
    "ModelEntry",
    "PreLaunchChecklist",
    "ChampionChallengerFramework",
    "format_governance_report",
]


# ---------------------------------------------------------------------------
# 枚举定义
# ---------------------------------------------------------------------------

class ModelRole(Enum):
    """模型在Champion Challenger框架中的角色。"""

    CHAMPION = "champion"  # 当前production模型
    CHALLENGER = "challenger"  # 候选新模型
    GATEKEEPER = "gatekeeper"  # 必须跑赢的baseline
    SHADOW = "shadow"  # 仅在shadow环境运行


class DegradationTrigger(Enum):
    """模型降级触发条件类型。"""

    PERFORMANCE = "performance"  # 连续3个月IR<0.3
    REGIME_MISMATCH = "regime"  # 当前regime下历史IC为负
    FACTOR_DECAY = "factor_decay"  # 核心因子IC 6个月下降>50%
    MARKET_EVENT = "market_event"  # 重大市场事件


# ---------------------------------------------------------------------------
# 数据类定义
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    """
    模型注册表条目。

    Attributes
    ----------
    name : str
        模型唯一名称。
    role : ModelRole
        模型角色。
    model_type : str
        "linear" / "nonlinear" / "ltr"。
    status : str
        "active" / "shadow" / "degraded" / "retired"。
    ir : float
        Information Ratio（年化）。
    sharpe : float
        Sharpe Ratio（年化）。
    max_dd : float
        最大回撤。
    hit_rate : float
        命中率（方向判断正确率）。
    turnover : float
        换手率。
    shadow_since : Optional[str]
        Shadow模式开始日期（ISO格式）。
    last_review : Optional[str]
        上次评审日期（ISO格式）。
    metadata : Dict[str, Any]
        额外元数据。
    """

    name: str
    role: ModelRole
    model_type: str  # "linear" / "nonlinear" / "ltr"
    status: str  # "active" / "shadow" / "degraded" / "retired"
    ir: float
    sharpe: float
    max_dd: float
    hit_rate: float
    turnover: float
    shadow_since: Optional[str] = None
    last_review: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        d = asdict(self)
        d["role"] = self.role.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelEntry":
        """从字典反序列化。"""
        d = dict(d)
        d["role"] = ModelRole(d["role"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PreLaunchChecklist:
    """
    模型上线前检验清单。

    涵盖五个维度的17项检验，确保模型符合机构级上线标准。

    Attributes
    ----------
    economic_sense : bool
        因子是否符合经济直觉。
    shap_ic_aligned : bool
        SHAP与IC方向一致。
    no_black_box : bool
        无无法解释的黑箱信号。
    ir_above_threshold : bool
        IR > 0.5。
    oos_r2_positive : bool
        OOS R^2 > 0。
    purged_cv_ok : bool
        Purged CV与naive CV差异<20%。
    time_stability : bool
        不同时窗performance一致。
    cross_market_consistency : bool
        至少3个市场方向一致。
    parameter_robustness : bool
        参数变化performance变化<10%。
    factor_dropout_robust : bool
        去掉1个因子performance下降<15%。
    point_in_time_data : bool
        使用PIT数据。
    no_survivorship_bias : bool
        包含退市股票。
    transaction_costs : bool
        考虑交易成本。
    purged_cv_used : bool
        使用Purged CV。
    beat_equal_weighted : bool
        跑赢等权baseline。
    beat_ic_weighted : bool
        跑赢IC加权baseline。
    oos_outperformance : bool
        OOS期间跑赢。
    """

    # A. 经济直觉检验
    economic_sense: bool = False
    shap_ic_aligned: bool = False
    no_black_box: bool = False

    # B. 统计显著性
    ir_above_threshold: bool = False
    oos_r2_positive: bool = False
    purged_cv_ok: bool = False

    # C. 稳健性
    time_stability: bool = False
    cross_market_consistency: bool = False
    parameter_robustness: bool = False
    factor_dropout_robust: bool = False

    # D. 回测合规
    point_in_time_data: bool = False
    no_survivorship_bias: bool = False
    transaction_costs: bool = False
    purged_cv_used: bool = False

    # E. Benchmark跑赢
    beat_equal_weighted: bool = False
    beat_ic_weighted: bool = False
    oos_outperformance: bool = False

    # 内部分类映射
    _CATEGORIES = {
        "A. Economic Sense": [
            "economic_sense",
            "shap_ic_aligned",
            "no_black_box",
        ],
        "B. Statistical Significance": [
            "ir_above_threshold",
            "oos_r2_positive",
            "purged_cv_ok",
        ],
        "C. Robustness": [
            "time_stability",
            "cross_market_consistency",
            "parameter_robustness",
            "factor_dropout_robust",
        ],
        "D. Backtest Compliance": [
            "point_in_time_data",
            "no_survivorship_bias",
            "transaction_costs",
            "purged_cv_used",
        ],
        "E. Benchmark": [
            "beat_equal_weighted",
            "beat_ic_weighted",
            "oos_outperformance",
        ],
    }

    def all_passed(self) -> bool:
        """检查所有17项检验是否全部通过。"""
        return all(
            [
                self.economic_sense,
                self.shap_ic_aligned,
                self.no_black_box,
                self.ir_above_threshold,
                self.oos_r2_positive,
                self.purged_cv_ok,
                self.time_stability,
                self.cross_market_consistency,
                self.parameter_robustness,
                self.factor_dropout_robust,
                self.point_in_time_data,
                self.no_survivorship_bias,
                self.transaction_costs,
                self.purged_cv_used,
                self.beat_equal_weighted,
                self.beat_ic_weighted,
                self.oos_outperformance,
            ]
        )

    def summary(self) -> Dict[str, Any]:
        """
        返回检验结果摘要。

        Returns
        -------
        dict
            {
                total: 17,
                passed: X,
                failed: Y,
                pass_rate: float,
                categories: {category: {passed: X, total: Y, items: {...}}}
            }
        """
        all_fields = [
            f for fields in self._CATEGORIES.values() for f in fields
        ]
        passed = sum(getattr(self, f) for f in all_fields)
        total = len(all_fields)

        categories = {}
        for cat_name, fields in self._CATEGORIES.items():
            cat_passed = sum(getattr(self, f) for f in fields)
            cat_items = {
                f: getattr(self, f) for f in fields
            }
            categories[cat_name] = {
                "passed": cat_passed,
                "total": len(fields),
                "pass_rate": round(cat_passed / len(fields), 2),
                "items": cat_items,
            }

        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 4) if total > 0 else 0.0,
            "all_passed": passed == total,
            "categories": categories,
        }

    def to_dict(self) -> Dict[str, bool]:
        """序列化为字典。"""
        return {
            "economic_sense": self.economic_sense,
            "shap_ic_aligned": self.shap_ic_aligned,
            "no_black_box": self.no_black_box,
            "ir_above_threshold": self.ir_above_threshold,
            "oos_r2_positive": self.oos_r2_positive,
            "purged_cv_ok": self.purged_cv_ok,
            "time_stability": self.time_stability,
            "cross_market_consistency": self.cross_market_consistency,
            "parameter_robustness": self.parameter_robustness,
            "factor_dropout_robust": self.factor_dropout_robust,
            "point_in_time_data": self.point_in_time_data,
            "no_survivorship_bias": self.no_survivorship_bias,
            "transaction_costs": self.transaction_costs,
            "purged_cv_used": self.purged_cv_used,
            "beat_equal_weighted": self.beat_equal_weighted,
            "beat_ic_weighted": self.beat_ic_weighted,
            "oos_outperformance": self.oos_outperformance,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, bool]) -> "PreLaunchChecklist":
        """从字典反序列化。"""
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Champion Challenger 框架主控类
# ---------------------------------------------------------------------------

class ChampionChallengerFramework:
    """
    Champion Challenger框架主控类。

    管理机构级模型的全生命周期：注册、shadow评估、挑战、降级。

    Attributes
    ----------
    models : Dict[str, ModelEntry]
        模型注册表。
    checklists : Dict[str, PreLaunchChecklist]
        模型上线前检验清单。
    shadow_logs : Dict[str, List[Dict]]
        Shadow运行日志。
    alerts : List[Dict]
        系统告警列表。
    """

    # 降级阈值
    DEGRADATION_THRESHOLDS = {
        "min_ir": 0.3,
        "min_ir_months": 3,
        "factor_ic_decay_threshold": 0.5,
        "factor_ic_decay_months": 6,
    }

    def __init__(self) -> None:
        self.models: Dict[str, ModelEntry] = {}
        self.checklists: Dict[str, PreLaunchChecklist] = {}
        self.shadow_logs: Dict[str, List[Dict[str, Any]]] = {}
        self.alerts: List[Dict[str, Any]] = []
        self._degradation_history: Dict[str, List[Dict]] = {}

    # ---- 模型注册与管理 ----

    def register_model(self, entry: ModelEntry) -> None:
        """
        注册模型到框架。

        Parameters
        ----------
        entry : ModelEntry
            模型注册条目。

        Raises
        ------
        ValueError
            同名模型已存在。
        """
        if entry.name in self.models:
            raise ValueError(f"Model '{entry.name}' already registered")
        self.models[entry.name] = entry
        self.checklists[entry.name] = PreLaunchChecklist()
        self.shadow_logs[entry.name] = []
        self._degradation_history[entry.name] = []

    def unregister_model(self, name: str) -> None:
        """注销模型。"""
        for d in [self.models, self.checklists, self.shadow_logs, self._degradation_history]:
            d.pop(name, None)

    def update_model_metrics(self, name: str, metrics: Dict[str, float]) -> None:
        """更新模型指标。"""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found")
        entry = self.models[name]
        for key in ["ir", "sharpe", "max_dd", "hit_rate", "turnover"]:
            if key in metrics:
                setattr(entry, key, metrics[key])
        entry.last_review = datetime.now().isoformat()

    # ---- 上线前检验 ----

    def run_prelaunch_checklist(
        self,
        model_name: str,
        checklist: Optional[PreLaunchChecklist] = None,
    ) -> PreLaunchChecklist:
        """
        运行/更新上线前检验清单。

        Parameters
        ----------
        model_name : str
            模型名称。
        checklist : Optional[PreLaunchChecklist]
            外部提供的检验结果。None时返回当前清单。

        Returns
        -------
        PreLaunchChecklist
            检验清单对象。
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        if checklist is not None:
            self.checklists[model_name] = checklist

        return self.checklists[model_name]

    # ---- Shadow Period ----

    def log_shadow_result(
        self, model_name: str, period: str, metrics: Dict[str, float]
    ) -> None:
        """
        记录shadow运行结果。

        Parameters
        ----------
        model_name : str
            模型名称。
        period : str
            运行期间（如 "2024-01"）。
        metrics : dict
            {ir, sharpe, hit_rate, ...}。
        """
        if model_name not in self.shadow_logs:
            self.shadow_logs[model_name] = []
        self.shadow_logs[model_name].append(
            {"period": period, **metrics}
        )

    def evaluate_shadow_period(
        self, model_name: str, min_months: int = 6
    ) -> bool:
        """
        评估shadow period表现，决定是否可挑战champion。

        条件：
        1. Shadow运行 >= min_months
        2. 平均IR > 0.5
        3. 通过率 > 60%
        4. 无连续2个月IR < 0

        Parameters
        ----------
        model_name : str
            模型名称。
        min_months : int, default 6
            最低shadow月数。

        Returns
        -------
        bool
            是否通过shadow评估。
        """
        logs = self.shadow_logs.get(model_name, [])
        if len(logs) < min_months:
            return False

        irs = [log.get("ir", 0) for log in logs]
        avg_ir = np.mean(irs)
        if avg_ir < 0.5:
            return False

        # 通过率
        positive_months = sum(1 for ir in irs if ir > 0)
        if positive_months / len(irs) < 0.6:
            return False

        # 无连续2个月IR < 0
        consecutive_negative = 0
        for ir in irs:
            if ir < 0:
                consecutive_negative += 1
                if consecutive_negative >= 2:
                    return False
            else:
                consecutive_negative = 0

        return True

    # ---- Champion挑战 ----

    def challenge_champion(self, challenger_name: str) -> bool:
        """
        Challenger尝试替换Champion。

        必须同时满足：
        1. Challenger的shadow评估通过
        2. Challenger IR > Champion IR
        3. Challenger跑赢等权gatekeeper
        4. Challenger跑赢IC加权gatekeeper

        Parameters
        ----------
        challenger_name : str
            挑战者模型名称。

        Returns
        -------
        bool
            是否成功替换Champion。
        """
        if challenger_name not in self.models:
            raise ValueError(f"Model '{challenger_name}' not found")

        challenger = self.models[challenger_name]
        if challenger.role != ModelRole.CHALLENGER:
            raise ValueError(
                f"Model '{challenger_name}' is not a challenger"
            )

        # 1. Shadow评估
        if not self.evaluate_shadow_period(challenger_name):
            self.alerts.append(
                {
                    "type": "challenge_failed",
                    "model": challenger_name,
                    "reason": "Shadow period evaluation not passed",
                }
            )
            return False

        # 2. 找到当前Champion
        champion = None
        for entry in self.models.values():
            if entry.role == ModelRole.CHAMPION:
                champion = entry
                break

        if champion is None:
            # 无当前champion，直接晋升
            challenger.role = ModelRole.CHAMPION
            challenger.status = "active"
            return True

        # 3. IR比较
        if challenger.ir <= champion.ir:
            self.alerts.append(
                {
                    "type": "challenge_failed",
                    "model": challenger_name,
                    "reason": f"Challenger IR ({challenger.ir:.3f}) <= Champion IR ({champion.ir:.3f})",
                }
            )
            return False

        # 4. 跑赢gatekeepers
        gatekeepers = [
            e for e in self.models.values()
            if e.role == ModelRole.GATEKEEPER
        ]
        for gk in gatekeepers:
            if challenger.ir <= gk.ir:
                self.alerts.append(
                    {
                        "type": "challenge_failed",
                        "model": challenger_name,
                        "reason": f"Challenger IR ({challenger.ir:.3f}) <= Gatekeeper '{gk.name}' IR ({gk.ir:.3f})",
                    }
                )
                return False

        # 成功替换
        champion.role = ModelRole.SHADOW
        champion.status = "shadow"
        challenger.role = ModelRole.CHAMPION
        challenger.status = "active"

        self.alerts.append(
            {
                "type": "champion_changed",
                "old_champion": champion.name,
                "new_champion": challenger_name,
            }
        )
        return True

    # ---- 降级检测与处理 ----

    def check_degradation_triggers(
        self, model_name: str, recent_metrics: Dict[str, Any]
    ) -> List[DegradationTrigger]:
        """
        检查模型是否触发降级条件。

        Parameters
        ----------
        model_name : str
            模型名称。
        recent_metrics : dict
            近期指标，包含:
            - monthly_ir: List[float] 近N月IR
            - current_regime: str 当前regime
            - regime_ic: Dict[str, float] 当前regime下各因子IC
            - factor_ic_history: Dict[str, List[float]] 因子IC历史

        Returns
        -------
        List[DegradationTrigger]
            触发的降级条件列表。
        """
        triggers: List[DegradationTrigger] = []

        # 1. Performance trigger: 连续3个月IR < 0.3
        monthly_ir = recent_metrics.get("monthly_ir", [])
        if len(monthly_ir) >= self.DEGRADATION_THRESHOLDS["min_ir_months"]:
            recent_irs = monthly_ir[-self.DEGRADATION_THRESHOLDS["min_ir_months"]:]
            if all(ir < self.DEGRADATION_THRESHOLDS["min_ir"] for ir in recent_irs):
                triggers.append(DegradationTrigger.PERFORMANCE)

        # 2. Regime mismatch: 当前regime下核心因子IC为负
        current_regime = recent_metrics.get("current_regime", "")
        regime_ic = recent_metrics.get("regime_ic", {})
        negative_core_factors = sum(
            1 for ic in regime_ic.values() if ic < 0
        )
        if negative_core_factors > len(regime_ic) * 0.5:
            triggers.append(DegradationTrigger.REGIME_MISMATCH)

        # 3. Factor decay: 核心因子IC 6个月下降>50%
        factor_ic_history = recent_metrics.get("factor_ic_history", {})
        for factor, ic_series in factor_ic_history.items():
            if len(ic_series) >= self.DEGRADATION_THRESHOLDS["factor_ic_decay_months"]:
                old_ic = abs(np.mean(ic_series[:3]))
                new_ic = abs(np.mean(ic_series[-3:]))
                if old_ic > 0 and new_ic / old_ic < (1 - self.DEGRADATION_THRESHOLDS["factor_ic_decay_threshold"]):
                    triggers.append(DegradationTrigger.FACTOR_DECAY)
                    break

        # 4. Market event: 预留接口
        if recent_metrics.get("market_event_trigger", False):
            triggers.append(DegradationTrigger.MARKET_EVENT)

        return triggers

    def downgrade_model(
        self, model_name: str, triggers: List[DegradationTrigger]
    ) -> None:
        """
        降级模型：切换至shadow、冻结使用、启动root cause分析。

        Parameters
        ----------
        model_name : str
            模型名称。
        triggers : List[DegradationTrigger]
            触发的降级条件。
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        entry = self.models[model_name]

        # 记录降级历史
        downgrade_record = {
            "timestamp": datetime.now().isoformat(),
            "previous_status": entry.status,
            "previous_role": entry.role.value,
            "triggers": [t.value for t in triggers],
        }
        self._degradation_history[model_name].append(downgrade_record)

        # 降级操作
        entry.status = "degraded"
        if entry.role == ModelRole.CHAMPION:
            # Champion降级时找backup
            backup = self._find_backup_champion()
            if backup:
                backup.role = ModelRole.CHAMPION
                backup.status = "active"
                self.alerts.append(
                    {
                        "type": "champion_backup_activated",
                        "degraded_model": model_name,
                        "backup_model": backup.name,
                        "triggers": [t.value for t in triggers],
                    }
                )
            else:
                self.alerts.append(
                    {
                        "type": "no_champion_backup",
                        "degraded_model": model_name,
                        "triggers": [t.value for t in triggers],
                    }
                )

        entry.role = ModelRole.SHADOW

        self.alerts.append(
            {
                "type": "model_downgraded",
                "model": model_name,
                "triggers": [t.value for t in triggers],
            }
        )

    def _find_backup_champion(self) -> Optional[ModelEntry]:
        """寻找最佳backup champion（IR最高的active challenger）。"""
        candidates = [
            e
            for e in self.models.values()
            if e.role == ModelRole.CHALLENGER and e.status == "active"
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.ir)

    # ---- Dashboard ----

    def get_dashboard(self) -> Dict[str, Any]:
        """
        返回模型治理dashboard数据。

        Returns
        -------
        dict
            {
                champion: ModelEntry | None,
                challengers: List[ModelEntry],
                gatekeepers: List[ModelEntry],
                shadows: List[ModelEntry],
                alerts: List[dict],
                summary: {total_models, active, degraded, retired}
            }
        """
        champion = None
        challengers, gatekeepers, shadows = [], [], []

        for entry in self.models.values():
            if entry.role == ModelRole.CHAMPION:
                champion = entry
            elif entry.role == ModelRole.CHALLENGER:
                challengers.append(entry)
            elif entry.role == ModelRole.GATEKEEPER:
                gatekeepers.append(entry)
            elif entry.role == ModelRole.SHADOW:
                shadows.append(entry)

        active_count = sum(
            1 for e in self.models.values() if e.status == "active"
        )
        degraded_count = sum(
            1 for e in self.models.values() if e.status == "degraded"
        )
        retired_count = sum(
            1 for e in self.models.values() if e.status == "retired"
        )

        return {
            "champion": champion.to_dict() if champion else None,
            "challengers": [e.to_dict() for e in challengers],
            "gatekeepers": [e.to_dict() for e in gatekeepers],
            "shadows": [e.to_dict() for e in shadows],
            "alerts": self.alerts[-20:],  # 最近20条
            "summary": {
                "total_models": len(self.models),
                "active": active_count,
                "degraded": degraded_count,
                "retired": retired_count,
            },
        }

    # ---- 持久化 ----

    def save(self, filepath: str) -> None:
        """保存框架状态到JSON。"""
        state = {
            "models": {
                name: entry.to_dict()
                for name, entry in self.models.items()
            },
            "checklists": {
                name: checklist.to_dict()
                for name, checklist in self.checklists.items()
            },
            "shadow_logs": self.shadow_logs,
            "alerts": self.alerts,
            "degradation_history": self._degradation_history,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def load(self, filepath: str) -> None:
        """从JSON加载框架状态。"""
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)

        self.models = {
            name: ModelEntry.from_dict(d)
            for name, d in state.get("models", {}).items()
        }
        self.checklists = {
            name: PreLaunchChecklist.from_dict(d)
            for name, d in state.get("checklists", {}).items()
        }
        self.shadow_logs = state.get("shadow_logs", {})
        self.alerts = state.get("alerts", [])
        self._degradation_history = state.get(
            "degradation_history", {}
        )


# ---------------------------------------------------------------------------
# 报告格式化
# ---------------------------------------------------------------------------

def format_governance_report(framework: ChampionChallengerFramework) -> str:
    """
    格式化治理报告为可读字符串。

    Parameters
    ----------
    framework : ChampionChallengerFramework
        治理框架实例。

    Returns
    -------
    str
        格式化报告。
    """
    dash = framework.get_dashboard()
    lines = []
    lines.append("=" * 70)
    lines.append(" Mini-GRP v3.1 — Model Governance Report")
    lines.append("=" * 70)

    # Summary
    s = dash["summary"]
    lines.append(f"\n[Summary] Total: {s['total_models']} | Active: {s['active']} | Degraded: {s['degraded']} | Retired: {s['retired']}")

    # Champion
    champ = dash["champion"]
    if champ:
        lines.append(f"\n[CHAMPION] {champ['name']} (IR={champ['ir']:.3f}, Sharpe={champ['sharpe']:.3f})")
    else:
        lines.append("\n[CHAMPION] None")

    # Gatekeepers
    lines.append(f"\n[GATEKEEPERS] ({len(dash['gatekeepers'])})")
    for gk in dash["gatekeepers"]:
        lines.append(f"  - {gk['name']}: IR={gk['ir']:.3f}")

    # Challengers
    lines.append(f"\n[CHALLENGERS] ({len(dash['challengers'])})")
    for ch in dash["challengers"]:
        lines.append(f"  - {ch['name']}: IR={ch['ir']:.3f}, Status={ch['status']}")

    # Alerts
    if dash["alerts"]:
        lines.append(f"\n[ALERTS] ({len(dash['alerts'])} recent)")
        for alert in dash["alerts"][-5:]:
            lines.append(f"  [{alert.get('type', 'unknown')}] {alert}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 用法示例
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 创建治理框架
    framework = ChampionChallengerFramework()

    # 注册Gatekeeper Baselines
    framework.register_model(
        ModelEntry(
            name="EqualWeighted",
            role=ModelRole.GATEKEEPER,
            model_type="linear",
            status="active",
            ir=0.45,
            sharpe=0.80,
            max_dd=-0.15,
            hit_rate=0.52,
            turnover=0.30,
        )
    )
    framework.register_model(
        ModelEntry(
            name="ICWeighted",
            role=ModelRole.GATEKEEPER,
            model_type="linear",
            status="active",
            ir=0.55,
            sharpe=0.95,
            max_dd=-0.12,
            hit_rate=0.55,
            turnover=0.35,
        )
    )

    # 注册Champion
    framework.register_model(
        ModelEntry(
            name="ElasticNet_v2",
            role=ModelRole.CHAMPION,
            model_type="linear",
            status="active",
            ir=0.72,
            sharpe=1.10,
            max_dd=-0.10,
            hit_rate=0.58,
            turnover=0.45,
        )
    )

    # 注册Challenger
    framework.register_model(
        ModelEntry(
            name="LightGBM_v1",
            role=ModelRole.CHALLENGER,
            model_type="nonlinear",
            status="shadow",
            ir=0.85,
            sharpe=1.25,
            max_dd=-0.08,
            hit_rate=0.62,
            turnover=0.60,
            shadow_since="2024-01-01",
        )
    )

    # 模拟shadow日志
    for month in range(1, 7):
        framework.log_shadow_result(
            "LightGBM_v1",
            f"2024-{month:02d}",
            {"ir": 0.6 + month * 0.05, "sharpe": 1.0 + month * 0.05},
        )

    # 运行预上线检查
    checklist = PreLaunchChecklist(
        economic_sense=True,
        shap_ic_aligned=True,
        no_black_box=True,
        ir_above_threshold=True,
        oos_r2_positive=True,
        purged_cv_ok=True,
        time_stability=True,
        cross_market_consistency=True,
        parameter_robustness=True,
        factor_dropout_robust=True,
        point_in_time_data=True,
        no_survivorship_bias=True,
        transaction_costs=True,
        purged_cv_used=True,
        beat_equal_weighted=True,
        beat_ic_weighted=True,
        oos_outperformance=True,
    )
    framework.run_prelaunch_checklist("LightGBM_v1", checklist)
    print(f"Checklist all passed: {checklist.all_passed()}")
    print(f"Checklist summary: {checklist.summary()}")

    # 挑战Champion
    success = framework.challenge_champion("LightGBM_v1")
    print(f"\nChallenge result: {'SUCCESS' if success else 'FAILED'}")

    # 打印报告
    print(format_governance_report(framework))

    # 测试降级
    triggers = framework.check_degradation_triggers(
        "ElasticNet_v2",
        {
            "monthly_ir": [0.20, 0.15, 0.10],
            "current_regime": "Bear",
            "regime_ic": {"momentum": -0.05, "value": -0.02},
        },
    )
    print(f"\nDegradation triggers for ElasticNet_v2: {[t.value for t in triggers]}")
