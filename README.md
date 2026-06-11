# Mini-GRP v3.2 — 量化选股与回测系统

> **设计参考**: Principal Global Investors GRP (Global Research Platform)

Mini-GRP 是一个数据驱动的量化选股系统，支持 A 股、美股、港股等多市场，集成 19 个核心因子、5 维度评分体系、ML 增强与 Walk-forward 回测框架。

---

## 目录结构

```
mini_grp_v32/
├── README.md                          # 项目说明
├── requirements.txt                   # 依赖清单
├── app.py                             # Streamlit 应用入口（预留）
├── config/                            # 配置与模型治理
│   ├── __init__.py
│   └── settings.py                    # Champion-Challenger 框架、模型治理
├── core/                              # 核心引擎
│   ├── __init__.py
│   ├── factor_engine.py              # 19 因子计算
│   ├── scoring_engine.py             # 5 维度评分
│   └── main.py                       # CLI 主入口
├── data/                              # 数据层
│   ├── __init__.py
│   ├── data_engine.py                # 旧版 akshare/yfinance
│   ├── tushare_engine.py            # Tushare Pro (A股)
│   ├── alpha_vantage_engine.py       # Alpha Vantage (美股)
│   └── unified_fetcher.py           # 统一数据接口 + 本地缓存
├── backtest/                          # 回测层
│   ├── __init__.py
│   └── engine.py                     # Walk-forward 回测
├── ml/                                # ML 增强
│   ├── __init__.py
│   ├── selector.py                   # XGBoost 因子选择
│   ├── nonlinear_scorer.py          # 非线性评分
│   ├── regime_detector.py           # HMM 市场环境检测
│   ├── bandit.py                    # MAB 动态权重
│   └── validator.py                # 因子验证
├── analytics/                         # 分析工具
│   ├── __init__.py
│   ├── diversifier.py              # 相关性分散
│   ├── distance_corr.py            # 距离相关
│   ├── monte_carlo.py              # MC 风险分析
│   ├── voi.py                      # 信息价值
│   └── benchmark.py                # 基准模型
├── viz/                               # 可视化
│   ├── __init__.py
│   └── visualizer.py               # 报告生成与图表
├── utils/                             # 工具
│   ├── __init__.py
│   └── mock.py                     # Mock 数据生成
├── tests/                             # 测试（预留）
│   └── __init__.py
├── cache/                             # 运行时缓存
└── output/                            # 运行时输出
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

在环境变量或 `.env` 文件中配置数据源 API Key：

```bash
export TUSHARE_TOKEN="your_tushare_token"
export ALPHA_VANTAGE_API_KEY="your_av_key"
```

### 3. 运行命令

**A 股实时选股（默认）**
```bash
python -m core.main --market cn
```

**美股实时选股**
```bash
python -m core.main --market us --max-stocks 100
```

**A 股回测（2020-2024）**
```bash
python -m core.main --market cn --backtest --start-date 2020-01-01 --end-date 2024-12-31
```

**跨市场选股**
```bash
python -m core.main --market cn us --max-stocks-per-market 50
```

**启用 ML 增强**
```bash
python -m core.main --market cn --use-ml
```

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **19 因子** | Value / Quality / Growth / Momentum / Expectation 五维度共 19 个核心因子 |
| **5 维度评分** | 价值、质量、成长、动量、预期差距，支持行业内排名与复合得分 |
| **预期差距** | 接入分析师预期数据（SUE、EPS 修正、评级修正），捕捉预期差 |
| **ML 增强** | XGBoost 因子选择、非线性评分、HMM 市场环境检测、MAB 动态权重 |
| **回测框架** | Walk-forward 回测，支持月/季调仓、交易成本、绩效分析 |
| **模型治理** | Champion-Challenger 框架、17 项上线检验清单、降级触发机制 |
| **多市场** | A 股 (Tushare)、美股 (Alpha Vantage)、港股/日股/韩股 (yfinance) |
| **统一数据** | 统一接口 + 本地缓存，自动降级到旧版接口或 Mock 数据 |

---

## 文件分类说明

| 目录 | 职责 | 关键文件 |
|------|------|----------|
| `core/` | 选股核心流程 | `factor_engine.py`, `scoring_engine.py`, `main.py` |
| `data/` | 数据获取与缓存 | `unified_fetcher.py`, `tushare_engine.py`, `alpha_vantage_engine.py` |
| `backtest/` | 策略回测与绩效分析 | `engine.py` |
| `ml/` | 机器学习增强 | `selector.py`, `nonlinear_scorer.py`, `regime_detector.py`, `bandit.py` |
| `analytics/` | 风险分析与基准 | `monte_carlo.py`, `diversifier.py`, `benchmark.py`, `voi.py` |
| `viz/` | 可视化与报告 | `visualizer.py` |
| `utils/` | 工具与 Mock | `mock.py` |
| `config/` | 配置与治理 | `settings.py` (模型治理、Champion-Challenger) |

---

## 版本历史

- **v3.2** — 目录结构重构，模块化整理
- **v3.1** — ML 增强（XGBoost、HMM、MAB）
- **v3.0** — 统一数据接口、多市场支持
- **v2.x** — 5 维度评分、Walk-forward 回测
- **v1.x** — 基础因子计算与线性评分

---

## 许可证

仅供研究与学习使用。
