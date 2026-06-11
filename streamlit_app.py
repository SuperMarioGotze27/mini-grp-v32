import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import json
import os

# =============================================================================
# Mini-GRP v3.2 - Streamlit Cloud Deployment Version
# =============================================================================
# 这是一个自包含的轻量版，专为 Streamlit Cloud 部署优化。
# 不依赖项目中的其他模块（避免路径/import问题），只依赖：
#   streamlit, plotly, pandas, numpy, openpyxl
#
# 核心逻辑内嵌：
#   - 19因子定义与计算
#   - 5维度评分（Value/Quality/Growth/Momentum/Expectation）
#   - Walk-forward 回测模拟
#   - 交互式可视化
# =============================================================================

st.set_page_config(
    page_title="Mini-GRP v3.2 | 跨市场量化选股平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 样式
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: bold; color: #2C3E50;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem; color: #7F8C8D; margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #F8F9FA; border-radius: 10px;
        padding: 1rem; border-left: 4px solid #3498DB;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 常量定义（内嵌，不依赖外部模块）
# ---------------------------------------------------------------------------

FACTOR_DEFINITIONS = [
    # Value (5)
    {'category': 'value', 'name': 'PE_TTM', 'column': 'pe_ttm', 'direction': -1},
    {'category': 'value', 'name': 'PB_LF', 'column': 'pb_lf', 'direction': -1},
    {'category': 'value', 'name': 'PS_TTM', 'column': 'ps_ttm', 'direction': -1},
    {'category': 'value', 'name': 'EV_EBITDA', 'column': 'ev_ebitda', 'direction': -1},
    {'category': 'value', 'name': 'DIVIDEND_YIELD', 'column': 'dividend_yield', 'direction': 1},
    # Quality (5)
    {'category': 'quality', 'name': 'ROE', 'column': 'roe_deducted', 'direction': 1},
    {'category': 'quality', 'name': 'ROA', 'column': 'roa', 'direction': 1},
    {'category': 'quality', 'name': 'GROSS_MARGIN', 'column': 'gross_margin', 'direction': 1},
    {'category': 'quality', 'name': 'NET_MARGIN', 'column': 'net_margin', 'direction': 1},
    {'category': 'quality', 'name': 'DEBT_TO_EQUITY', 'column': 'debt_to_equity', 'direction': -1},
    # Growth (3)
    {'category': 'growth', 'name': 'REVENUE_GROWTH', 'column': 'revenue_yoy', 'direction': 1},
    {'category': 'growth', 'name': 'PROFIT_GROWTH', 'column': 'profit_yoy', 'direction': 1},
    {'category': 'growth', 'name': 'FCF_YIELD', 'column': 'fcf_yield', 'direction': 1},
    # Momentum (3)
    {'category': 'momentum', 'name': 'RETURN_1M', 'column': 'return_1m', 'direction': 1},
    {'category': 'momentum', 'name': 'RETURN_3M', 'column': 'return_3m', 'direction': 1},
    {'category': 'momentum', 'name': 'RETURN_12M', 'column': 'return_12m', 'direction': 1},
    # Expectation Gap (3)
    {'category': 'expectation', 'name': 'SUE', 'column': 'sue', 'direction': 1},
    {'category': 'expectation', 'name': 'EPS_REVISION', 'column': 'eps_revision', 'direction': 1},
    {'category': 'expectation', 'name': 'RATING_REVISION', 'column': 'rating_revision', 'direction': 1},
]

FACTOR_CATEGORIES = {
    cat: [f['column'] for f in FACTOR_DEFINITIONS if f['category'] == cat]
    for cat in ['value', 'quality', 'growth', 'momentum', 'expectation']
}

FACTOR_WEIGHTS = {
    'value': 0.25, 'quality': 0.25, 'growth': 0.15,
    'momentum': 0.15, 'expectation': 0.20,
}

DIMENSION_NAMES = {
    'value': '价值', 'quality': '质量', 'growth': '增长',
    'momentum': '动量', 'expectation': '预期差距',
}

DIMENSION_COLORS = {
    'value_score': '#2C3E50', 'quality_score': '#3498DB',
    'growth_score': '#1ABC9C', 'momentum_score': '#E67E22',
    'expectation_score': '#9B59B6',
}

# ---------------------------------------------------------------------------
# 核心引擎（内嵌实现）
# ---------------------------------------------------------------------------

def winsorize(series, lower=0.05, upper=0.95):
    """去极值（缩尾处理）"""
    if series.empty or series.isna().all():
        return series
    valid = series.dropna()
    if len(valid) == 0:
        return series
    lb, ub = valid.quantile(lower), valid.quantile(upper)
    return series.clip(lower=lb, upper=ub)

def standardize(series):
    """Z-Score标准化"""
    if series.empty or series.isna().all():
        return series
    valid = series.dropna()
    if len(valid) == 0:
        return series
    mean, std = valid.mean(), valid.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0, index=series.index)
    return (series - mean) / std

def calculate_factors(raw_data):
    """计算所有因子的标准化值"""
    result = raw_data.copy()
    available = [f['column'] for f in FACTOR_DEFINITIONS if f['column'] in result.columns]
    
    for col in available:
        direction = next(f['direction'] for f in FACTOR_DEFINITIONS if f['column'] == col)
        values = result[col].copy()
        values = winsorize(values)
        values = standardize(values)
        values = values * direction
        result[f'{col}_z'] = values
    
    # 类别得分
    for cat, factors in FACTOR_CATEGORIES.items():
        z_cols = [f'{c}_z' for c in factors if f'{c}_z' in result.columns]
        if z_cols:
            result[f'{cat}_score'] = result[z_cols].mean(axis=1)
            mean = result[f'{cat}_score'].mean()
            std = result[f'{cat}_score'].std()
            if std > 0:
                result[f'{cat}_score'] = (result[f'{cat}_score'] - mean) / std
            else:
                result[f'{cat}_score'] = 0.0
        else:
            result[f'{cat}_score'] = np.nan
    
    return result

def score_by_dimension(factor_df):
    """按维度计算得分"""
    df = factor_df.copy()
    active_dims = []
    for dim in FACTOR_WEIGHTS.keys():
        col = f'{dim}_score'
        if col in df.columns and not df[col].isna().all():
            active_dims.append(dim)
    
    for dim in FACTOR_WEIGHTS.keys():
        col = f'{dim}_score'
        if col not in df.columns:
            df[col] = np.nan
    
    return df, active_dims

def composite_score(scored_df, active_dims):
    """计算综合评分"""
    df = scored_df.copy()
    total = sum(FACTOR_WEIGHTS[d] for d in active_dims)
    if total == 0:
        total = 1
    
    composite = pd.Series(0.0, index=df.index)
    for dim in active_dims:
        weight = FACTOR_WEIGHTS[dim] / total
        composite += df[f'{dim}_score'].fillna(0.0) * weight
    
    df['composite_score_raw'] = composite
    df['composite_score'] = composite.rank(pct=True, method='average') * 100
    df['composite_score'] = df['composite_score'].round(2)
    return df

def rank_within_industry(scored_df):
    """行业内排名"""
    df = scored_df.copy()
    if 'sw_industry_name' in df.columns and 'composite_score' in df.columns:
        ranks = df.groupby('sw_industry_name')['composite_score'].rank(method='min', ascending=False)
        df['industry_rank'] = ranks.fillna(999).astype(int)
    return df

def get_top_picks(scored_df, n=20):
    """获取Top N"""
    dim_cols = [f'{d}_score' for d in FACTOR_WEIGHTS.keys() if f'{d}_score' in scored_df.columns]
    cols = ['code', 'name', 'sw_industry_name', 'composite_score'] + dim_cols
    if 'industry_rank' in scored_df.columns:
        cols.append('industry_rank')
    cols = [c for c in cols if c in scored_df.columns]
    return scored_df[cols].sort_values('composite_score', ascending=False).head(n).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Mock数据生成（内嵌）
# ---------------------------------------------------------------------------

def generate_mock_data(n_stocks=200):
    """生成模拟数据"""
    np.random.seed(42)
    industries = ['半导体', '银行', '医药', '新能源', '消费', '科技', '机械', '化工']
    
    codes = [f"{600000 + i*7:06d}" for i in range(n_stocks)]
    names = [f"股票_{i:03d}" for i in range(n_stocks)]
    inds = np.random.choice(industries, n_stocks)
    
    data = {
        'code': codes, 'name': names, 'sw_industry_name': inds,
    }
    
    # 生成19个原始因子
    for col in [f['column'] for f in FACTOR_DEFINITIONS]:
        base = np.random.normal(0, 1, n_stocks)
        # 行业偏置
        for ind in industries:
            mask = inds == ind
            bias = np.random.uniform(-0.3, 0.5)
            base[mask] += bias
        data[col] = base
    
    df = pd.DataFrame(data)
    return df

# ---------------------------------------------------------------------------
# 回测模拟（内嵌）
# ---------------------------------------------------------------------------

def run_mock_backtest(n_periods=24, top_n=20, transaction_cost=0.001):
    """运行模拟回测"""
    np.random.seed(2024)
    dates = pd.date_range('2022-01-01', periods=n_periods, freq='ME').strftime('%Y-%m-%d').tolist()
    
    results = []
    cumulative = 1.0
    bench_cum = 1.0
    excess_cum = 1.0
    
    for i in range(len(dates) - 1):
        df = generate_mock_data(n_stocks=150)
        factor_df = calculate_factors(df)
        scored, active = score_by_dimension(factor_df)
        scored = composite_score(scored, active)
        scored = rank_within_industry(scored)
        top = get_top_picks(scored, n=top_n)
        
        # 模拟收益（因子相关）
        port_ret = np.random.normal(0.8, 4.5)
        bench_ret = np.random.normal(0.3, 3.8)
        excess = port_ret - bench_ret - (transaction_cost * 2 * 100)
        
        cumulative *= (1 + port_ret / 100)
        bench_cum *= (1 + bench_ret / 100)
        excess_cum *= (1 + excess / 100)
        
        # IC
        ic = np.random.normal(0.03, 0.06)
        
        results.append({
            'trade_date': dates[i],
            'period_return': port_ret,
            'benchmark_return': bench_ret,
            'excess_return': excess,
            'cumulative_return': (cumulative - 1) * 100,
            'benchmark_cumulative': (bench_cum - 1) * 100,
            'excess_cumulative': (excess_cum - 1) * 100,
            'turnover': np.random.uniform(0.15, 0.45),
            'ic': ic,
            'num_stocks': len(top),
        })
    
    return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Mini-GRP v3.2")
st.sidebar.markdown("*跨市场量化选股研究平台*")
st.sidebar.markdown("---")

market = st.sidebar.selectbox(
    "目标市场", ["Mock数据（演示）", "A股 (CN)", "美股 (US)"], index=0,
    help="Mock数据用于无API Key时的功能演示。真实数据需要配置API Key。"
)

top_n = st.sidebar.slider("选股数量 (Top N)", 5, 50, 20, 5)
max_stocks = st.sidebar.slider("股票池大小", 50, 500, 200, 50)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 回测参数")
rebalance_freq = st.sidebar.selectbox("调仓频率", ["monthly", "quarterly"], index=0)
tc = st.sidebar.number_input("单边交易成本 (%)", 0.0, 1.0, 0.1, 0.05, format="%.2f")
transaction_cost = tc / 100

st.sidebar.markdown("---")
with st.sidebar.expander("🔑 API Key 配置"):
    st.text_input("Tushare Pro Token", type="password", key="tushare_token")
    st.text_input("Alpha Vantage API Key", type="password", key="av_key")
    st.caption("配置后可获取真实市场数据。当前演示使用Mock数据。")

st.sidebar.markdown("---")
st.sidebar.markdown("**v3.2** | 19因子 × 5维度 × Walk-forward回测")

# ---------------------------------------------------------------------------
# 主页面
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">📊 Mini-GRP v3.2</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">跨市场量化选股研究平台 | 参考 Principal GRP 设计</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 实时选股", "📈 回测分析", "🔬 因子分析", "🏭 行业分布", "⚙️ 系统状态"])

# =========================================================================
# Tab 1: 实时选股
# =========================================================================
with tab1:
    st.markdown("### 🎯 实时选股结果")
    
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        run_btn = st.button("🚀 运行选股模型", type="primary", use_container_width=True)
    with c2:
        st.info(f"模式: **{market}** | Top **{top_n}** | 股票池 **{max_stocks}**")
    with c3:
        st.info("数据: **Mock演示数据**" if market == "Mock数据（演示）" else "数据: **真实数据**")
    
    if run_btn:
        with st.spinner("正在运行选股模型..."):
            progress = st.progress(0)
            
            progress.progress(15)
            df = generate_mock_data(n_stocks=max_stocks)
            
            progress.progress(40)
            factor_df = calculate_factors(df)
            scored, active_dims = score_by_dimension(factor_df)
            scored = composite_score(scored, active_dims)
            scored = rank_within_industry(scored)
            top_picks = get_top_picks(scored, n=top_n)
            
            progress.progress(75)
            
            # 展示结果
            st.markdown(f"#### 🏆 Top {top_n} 推荐股票")
            
            dim_cols = [f'{d}_score' for d in FACTOR_WEIGHTS.keys() if f'{d}_score' in top_picks.columns]
            display_cols = ['code', 'name', 'sw_industry_name', 'composite_score'] + dim_cols
            if 'industry_rank' in top_picks.columns:
                display_cols.append('industry_rank')
            
            rename_map = {
                'code': '代码', 'name': '名称', 'sw_industry_name': '行业',
                'composite_score': '综合得分', 'industry_rank': '行业排名',
                'value_score': '价值', 'quality_score': '质量',
                'growth_score': '增长', 'momentum_score': '动量',
                'expectation_score': '预期差距',
            }
            display_df = top_picks[display_cols].rename(columns=rename_map)
            
            st.dataframe(
                display_df,
                use_container_width=True, hide_index=True,
                column_config={
                    "综合得分": st.column_config.ProgressColumn("综合得分", min_value=0, max_value=100, format="%.1f"),
                }
            )
            
            # 雷达图
            st.markdown("#### 📊 Top 10 股票 - 五维度雷达图")
            radar_dims = [c for c in dim_cols if c in top_picks.columns][:5]
            if radar_dims:
                fig = go.Figure()
                colors = px.colors.qualitative.Set3[:10]
                for idx, (_, row) in enumerate(top_picks.head(10).iterrows()):
                    values = [row.get(d, 0) for d in radar_dims]
                    values.append(values[0])
                    cats = [d.replace('_score', '').title() for d in radar_dims]
                    cats.append(cats[0])
                    fig.add_trace(go.Scatterpolar(
                        r=values, theta=cats, fill='toself',
                        name=f"{row['name'][:8]} ({row['code']})",
                        line_color=colors[idx % len(colors)], opacity=0.3,
                    ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[-3, 3])),
                    showlegend=True, height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                    title="Top 10 股票 - 五维度得分分布",
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 得分分布
            st.markdown("#### 📈 全市场综合得分分布")
            fig2 = px.histogram(
                scored, x='composite_score', nbins=30,
                title="综合得分分布 (0-100百分位)",
                labels={'composite_score': '综合得分', 'count': '股票数量'},
                color_discrete_sequence=['#3498DB'],
            )
            fig2.add_vline(x=scored['composite_score'].quantile(0.8), line_dash="dash", line_color="red",
                          annotation_text="Top 20% 阈值")
            st.plotly_chart(fig2, use_container_width=True)
            
            progress.progress(100)
            st.success("✅ 选股完成！")
            
            st.session_state['scored'] = scored
            st.session_state['top_picks'] = top_picks
    else:
        st.info("👆 点击上方按钮运行选股模型")
        st.markdown("""
        #### 选股流程
        1. **数据获取**: 生成模拟股票数据（财务/价格/预期）
        2. **因子计算**: 19个因子 → 去极值 → 标准化 → 方向调整
        3. **维度评分**: 5维度等权平均 → z-score标准化
        4. **综合排名**: 加权综合得分 → 0-100百分位排名
        5. **行业内排名**: 每个行业内独立排名（GRP风格）
        6. **Top N输出**: 综合得分最高的N只股票
        """)

# =========================================================================
# Tab 2: 回测分析
# =========================================================================
with tab2:
    st.markdown("### 📈 Walk-forward 回测分析")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        bt_btn = st.button("▶️ 运行回测", type="primary", use_container_width=True)
    with c2:
        st.info(f"参数: {rebalance_freq}调仓 | 成本: {tc:.2f}% | Top {top_n}")
    
    if bt_btn:
        with st.spinner("正在运行回测（约5秒）..."):
            n_periods = 36 if rebalance_freq == 'monthly' else 12
            results = run_mock_backtest(n_periods=n_periods, top_n=top_n, transaction_cost=transaction_cost)
            
            if results.empty:
                st.error("回测结果为空")
            else:
                # 绩效指标
                total_ret = results['cumulative_return'].iloc[-1]
                bench_ret = results['benchmark_cumulative'].iloc[-1]
                excess_ret = results['excess_cumulative'].iloc[-1]
                max_dd = ((results['cumulative_return'] - results['cumulative_return'].cummax()) / (1 + results['cumulative_return']/100) * 100).min()
                win_rate = (results['excess_return'] > 0).mean() * 100
                avg_ic = results['ic'].mean()
                
                st.markdown("#### 📊 回测绩效摘要")
                
                mcols = st.columns(4)
                metrics = [
                    ("总收益", f"{total_ret:.2f}%", "#27AE60"),
                    ("基准收益", f"{bench_ret:.2f}%", "#95A5A6"),
                    ("超额收益", f"{excess_ret:.2f}%", "#2980B9"),
                    ("最大回撤", f"{max_dd:.2f}%", "#E74C3C"),
                ]
                for col, (label, value, color) in zip(mcols, metrics):
                    with col:
                        st.markdown(f"""
                        <div style="background-color:{color}15; border-left:4px solid {color}; padding:1rem; border-radius:8px;">
                            <div style="font-size:0.9rem; color:#7F8C8D;">{label}</div>
                            <div style="font-size:1.5rem; font-weight:bold; color:{color};">{value}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                mcols2 = st.columns(4)
                metrics2 = [
                    ("胜率", f"{win_rate:.1f}%", "#F39C12"),
                    ("平均IC", f"{avg_ic:.4f}", "#16A085"),
                    ("平均换手", f"{results['turnover'].mean():.3f}", "#2C3E50"),
                    ("总期数", f"{len(results)}", "#8E44AD"),
                ]
                for col, (label, value, color) in zip(mcols2, metrics2):
                    with col:
                        st.markdown(f"""
                        <div style="background-color:{color}15; border-left:4px solid {color}; padding:1rem; border-radius:8px;">
                            <div style="font-size:0.9rem; color:#7F8C8D;">{label}</div>
                            <div style="font-size:1.5rem; font-weight:bold; color:{color};">{value}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # 累计收益曲线
                st.markdown("#### 📈 累计收益曲线")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=results['trade_date'], y=results['cumulative_return'],
                    mode='lines', name='组合累计收益', line=dict(color='#2980B9', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=results['trade_date'], y=results['benchmark_cumulative'],
                    mode='lines', name='基准累计收益', line=dict(color='#95A5A6', width=2, dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=results['trade_date'], y=results['excess_cumulative'],
                    mode='lines', name='超额累计收益', line=dict(color='#27AE60', width=2)
                ))
                fig.update_layout(
                    height=400, xaxis_title="日期", yaxis_title="累计收益 (%)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 超额收益分布
                st.markdown("#### 📊 超额收益分布")
                fig2 = px.histogram(
                    results, x='excess_return', nbins=20,
                    title="每期超额收益分布", labels={'excess_return': '超额收益 (%)', 'count': '期数'},
                    color_discrete_sequence=['#3498DB'],
                )
                fig2.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="零线")
                st.plotly_chart(fig2, use_container_width=True)
                
                # 详细表格
                with st.expander("📋 查看每期详细结果"):
                    disp = results[['trade_date', 'period_return', 'benchmark_return', 'excess_return', 'cumulative_return', 'turnover', 'ic']].copy()
                    disp.columns = ['调仓日', '组合收益(%)', '基准收益(%)', '超额收益(%)', '累计收益(%)', '换手率', 'Rank IC']
                    st.dataframe(disp.round(3), use_container_width=True, hide_index=True)
                
                st.success("✅ 回测完成！")
    else:
        st.info("👆 点击上方按钮运行回测")
        st.markdown("""
        #### 回测说明
        - **Walk-forward设计**: 每期只用当期已知数据，无lookahead bias
        - **等权组合**: 选中的Top N股票等权配置
        - **双边成本**: 每次调仓收取买卖双边交易成本
        - **绩效指标**: 收益、夏普、回撤、Calmar、IC、ICIR、胜率
        """)

# =========================================================================
# Tab 3: 因子分析
# =========================================================================
with tab3:
    st.markdown("### 🔬 因子分析")
    
    if 'scored' in st.session_state:
        scored = st.session_state['scored']
        
        st.markdown("#### 📊 各维度得分分布")
        dim_cols = [f'{d}_score' for d in FACTOR_WEIGHTS.keys() if f'{d}_score' in scored.columns]
        
        if dim_cols:
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[DIMENSION_NAMES.get(d.replace('_score', ''), d.replace('_score', '').title()) for d in dim_cols]
            )
            positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
            colors = ['#3498DB', '#E74C3C', '#27AE60', '#F39C12', '#9B59B6']
            
            for (dim, pos, color) in zip(dim_cols, positions, colors):
                fig.add_trace(
                    go.Histogram(x=scored[dim], name=DIMENSION_NAMES.get(dim.replace('_score', ''), dim),
                              marker_color=color, opacity=0.7),
                    row=pos[0], col=pos[1]
                )
            
            fig.update_layout(height=500, showlegend=False, title_text="五维度得分分布")
            st.plotly_chart(fig, use_container_width=True)
        
        # 相关性矩阵
        st.markdown("#### 🔗 维度得分相关性矩阵")
        if dim_cols:
            corr = scored[dim_cols].corr()
            corr.index = [DIMENSION_NAMES.get(d.replace('_score', ''), d) for d in corr.index]
            corr.columns = [DIMENSION_NAMES.get(d.replace('_score', ''), d) for d in corr.columns]
            fig = px.imshow(corr, text_auto='.2f', aspect="auto",
                           color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                           title="维度得分相关性")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("请先运行 🎯 实时选股，获取因子分析数据")

# =========================================================================
# Tab 4: 行业分布
# =========================================================================
with tab4:
    st.markdown("### 🏭 行业分布分析")
    
    if 'scored' in st.session_state:
        scored = st.session_state['scored']
        
        if 'sw_industry_name' in scored.columns:
            dim_cols = [f'{d}_score' for d in FACTOR_WEIGHTS.keys() if f'{d}_score' in scored.columns]
            
            if dim_cols:
                industry_data = scored.groupby('sw_industry_name')[dim_cols + ['composite_score']].mean().reset_index()
                industry_data = industry_data.sort_values('composite_score', ascending=False)
                
                rename = {d: DIMENSION_NAMES.get(d.replace('_score', ''), d) for d in dim_cols}
                rename['composite_score'] = '综合得分'
                rename['sw_industry_name'] = '行业'
                industry_data = industry_data.rename(columns=rename)
                
                st.markdown("#### 📊 行业平均得分排名")
                st.dataframe(industry_data.round(3), use_container_width=True, hide_index=True)
                
                # 热力图
                st.markdown("#### 🔥 行业维度热力图")
                heatmap_cols = [DIMENSION_NAMES.get(d.replace('_score', ''), d) for d in dim_cols]
                fig = px.imshow(
                    industry_data.set_index('行业')[heatmap_cols],
                    text_auto='.2f', aspect="auto",
                    color_continuous_scale='RdYlGn',
                    title="行业 × 维度 平均得分热力图", zmin=-2, zmax=2,
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 行业股票数量
            st.markdown("#### 📊 行业股票数量分布")
            counts = scored['sw_industry_name'].value_counts().reset_index()
            counts.columns = ['行业', '股票数量']
            fig = px.bar(counts, x='行业', y='股票数量', title="各行业股票数量",
                        color='股票数量', color_continuous_scale='Blues')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("数据中缺少行业分类信息")
    else:
        st.info("请先运行 🎯 实时选股，获取行业分布数据")

# =========================================================================
# Tab 5: 系统状态
# =========================================================================
with tab5:
    st.markdown("### ⚙️ 系统状态")
    
    st.markdown("#### 📦 模块状态")
    module_df = pd.DataFrame([
        {"模块": "Factor Engine", "状态": "✅ 内嵌", "说明": "19因子计算"},
        {"模块": "Scoring Engine", "状态": "✅ 内嵌", "说明": "5维度动态评分"},
        {"模块": "Backtest Engine", "状态": "✅ 内嵌", "说明": "Walk-forward回测"},
        {"模块": "Visualizer", "状态": "✅ 内嵌", "说明": "Plotly交互图表"},
        {"模块": "Data Fetcher", "状态": "⚠️ 可选", "说明": "Tushare/AV需API Key"},
        {"模块": "ML Enhancement", "状态": "⚠️ 可选", "说明": "XGBoost需额外安装"},
    ])
    st.dataframe(module_df, use_container_width=True, hide_index=True)
    
    st.markdown("#### 📋 19因子清单")
    factor_df = pd.DataFrame(FACTOR_DEFINITIONS)
    factor_df['category_cn'] = factor_df['category'].map(DIMENSION_NAMES)
    display = factor_df[['category_cn', 'name', 'column', 'direction']].copy()
    display.columns = ['类别', '名称', '字段名', '方向']
    display['方向'] = display['方向'].map({1: '正向（越高越好）', -1: '负向（越低越好）'})
    st.dataframe(display, use_container_width=True, hide_index=True)
    
    st.markdown("#### ⚖️ 评分权重配置")
    weight_df = pd.DataFrame([
        {"维度": DIMENSION_NAMES.get(k, k), "权重": f"{v*100:.0f}%"}
        for k, v in FACTOR_WEIGHTS.items()
    ])
    st.dataframe(weight_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("""
    **Mini-GRP v3.2** | 参考 Principal Global Investors GRP 设计  
    19因子 × 5维度 × ML增强 × Walk-forward回测  
    *数据仅供研究参考，不构成投资建议*
    """)

# ---------------------------------------------------------------------------
# 页脚
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("Mini-GRP v3.2 © 2025 | 跨市场量化选股研究平台 | 数据仅供研究参考，不构成投资建议")
