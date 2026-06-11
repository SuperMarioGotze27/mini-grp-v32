import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# 确保项目根目录在路径中
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------
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
        font-size: 2.5rem;
        font-weight: bold;
        color: #2C3E50;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #7F8C8D;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #3498DB;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 模块加载状态
# ---------------------------------------------------------------------------
@st.cache_resource
def load_modules():
    """加载核心模块，返回状态字典。"""
    status = {}
    modules = {}
    
    try:
        from core.factor_engine import calculate_factors, FACTOR_DEFINITIONS, FACTOR_CATEGORIES
        modules['factor_engine'] = True
        status['factor_engine'] = ('✅', '19因子计算引擎')
    except Exception as e:
        modules['factor_engine'] = False
        status['factor_engine'] = ('❌', f'因子引擎: {e}')
    
    try:
        from core.scoring_engine import score_by_dimension, composite_score, rank_within_industry, get_top_picks, FACTOR_WEIGHTS
        modules['scoring_engine'] = True
        status['scoring_engine'] = ('✅', '5维度评分引擎')
    except Exception as e:
        modules['scoring_engine'] = False
        status['scoring_engine'] = ('❌', f'评分引擎: {e}')
    
    try:
        from backtest.engine import BacktestConfig, run_backtest, analyze_backtest_results, generate_backtest_report
        modules['backtest'] = True
        status['backtest'] = ('✅', 'Walk-forward回测引擎')
    except Exception as e:
        modules['backtest'] = False
        status['backtest'] = ('❌', f'回测引擎: {e}')
    
    try:
        from utils.mock import generate_mock_data
        modules['mock'] = True
        status['mock'] = ('✅', 'Mock数据生成器')
    except Exception as e:
        modules['mock'] = False
        status['mock'] = ('❌', f'Mock数据: {e}')
    
    try:
        from data.unified_fetcher import fetch_a_share_data, fetch_us_data
        modules['data_fetcher'] = True
        status['data_fetcher'] = ('✅', '统一数据接口')
    except Exception as e:
        modules['data_fetcher'] = False
        status['data_fetcher'] = ('❌', f'数据接口: {e}')
    
    try:
        from ml.selector import MLFactorSelector
        modules['ml'] = True
        status['ml'] = ('✅', 'ML增强模块 (XGBoost)')
    except Exception as e:
        modules['ml'] = False
        status['ml'] = ('⚠️', f'ML模块未安装: {e}')
    
    return modules, status

modules, module_status = load_modules()

# ---------------------------------------------------------------------------
# 侧边栏配置
# ---------------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ 系统配置")

# 市场选择
market = st.sidebar.selectbox(
    "目标市场",
    options=["A股 (CN)", "美股 (US)", "Mock数据（演示）"],
    index=2,  # 默认Mock，因为真实数据需要API Key
    help="选择数据来源市场。Mock数据用于无API Key时的演示。"
)

# 选股参数
top_n = st.sidebar.slider("选股数量 (Top N)", min_value=5, max_value=50, value=20, step=5)
max_stocks = st.sidebar.slider("股票池大小", min_value=50, max_value=500, value=200, step=50)

# 回测参数
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 回测参数")
rebalance_freq = st.sidebar.selectbox("调仓频率", ["monthly", "quarterly"], index=0)
transaction_cost = st.sidebar.number_input("单边交易成本 (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05, format="%.2f")
transaction_cost = transaction_cost / 100  # 转换为小数

# ML选项
use_ml = st.sidebar.checkbox("启用 ML 增强 (XGBoost)", value=False, disabled=not modules.get('ml', False))
if use_ml and not modules.get('ml', False):
    st.sidebar.warning("ML模块未安装，运行: `pip install xgboost scikit-learn`")

# API Key配置（折叠）
with st.sidebar.expander("🔑 API Key 配置 (可选)"):
    tushare_token = st.text_input("Tushare Pro Token", type="password", help="用于获取A股分析师预期数据")
    alpha_vantage_key = st.text_input("Alpha Vantage API Key", type="password", help="用于获取美股财务数据")
    if tushare_token:
        os.environ['TUSHARE_TOKEN'] = tushare_token
    if alpha_vantage_key:
        os.environ['ALPHA_VANTAGE_API_KEY'] = alpha_vantage_key

st.sidebar.markdown("---")
st.sidebar.markdown("**Mini-GRP v3.2**  
*参考 Principal GRP 设计*")

# ---------------------------------------------------------------------------
# 主页面标题
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">📊 Mini-GRP v3.2</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">跨市场量化选股研究平台 | 19因子 × 5维度 × ML增强 × Walk-forward回测</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 数据获取函数
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_data(market_choice, n_stocks, _use_real=False):
    """获取股票数据（带缓存）。"""
    if market_choice == "Mock数据（演示）" or not _use_real:
        from utils.mock import generate_mock_data
        df = generate_mock_data(n_stocks=n_stocks)
        # 添加预期差距因子
        np.random.seed(42)
        df['sue'] = np.random.normal(0, 1, len(df))
        df['eps_revision'] = np.random.normal(0, 0.5, len(df))
        df['rating_revision'] = np.random.normal(0, 0.3, len(df))
        return df, "mock"
    
    # 真实数据获取（需要API Key）
    try:
        if market_choice == "A股 (CN)":
            from data.unified_fetcher import fetch_a_share_data
            df = fetch_a_share_data(max_stocks=n_stocks, use_cache=True)
            return df, "real"
        elif market_choice == "美股 (US)":
            from data.unified_fetcher import fetch_us_data
            df = fetch_us_data(max_stocks=n_stocks, use_cache=True)
            return df, "real"
    except Exception as e:
        st.error(f"真实数据获取失败: {e}，回退到Mock数据")
        from utils.mock import generate_mock_data
        df = generate_mock_data(n_stocks=n_stocks)
        np.random.seed(42)
        df['sue'] = np.random.normal(0, 1, len(df))
        df['eps_revision'] = np.random.normal(0, 0.5, len(df))
        df['rating_revision'] = np.random.normal(0, 0.3, len(df))
        return df, "mock_fallback"
    
    return None, "error"

# ---------------------------------------------------------------------------
# 核心选股流程
# ---------------------------------------------------------------------------
def run_screening(df, use_ml=False):
    """运行选股流程。"""
    from core.factor_engine import calculate_factors
    from core.scoring_engine import score_by_dimension, composite_score, rank_within_industry, get_top_picks
    
    # 因子计算
    factor_df = calculate_factors(df)
    
    # 评分
    scored = score_by_dimension(factor_df)
    scored = composite_score(scored)
    scored = rank_within_industry(scored)
    
    # 获取Top N
    top_picks = get_top_picks(scored, n=top_n)
    
    return scored, top_picks

# ---------------------------------------------------------------------------
# Tab 1: 实时选股
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 实时选股", "📈 回测分析", "🔬 因子分析", "🏭 行业分布", "⚙️ 系统状态"
])

with tab1:
    st.markdown("### 🎯 实时选股结果")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        run_button = st.button("🚀 运行选股模型", type="primary", use_container_width=True)
    with col2:
        st.info(f"当前模式: **{market}** | Top **{top_n}** | 股票池 **{max_stocks}**")
    with col3:
        if use_ml:
            st.success("ML增强: 已启用")
        else:
            st.info("ML增强: 未启用")
    
    if run_button:
        with st.spinner("正在运行选股模型..."):
            progress_bar = st.progress(0)
            
            # 1. 获取数据
            progress_bar.progress(10)
            df, data_source = get_data(market, max_stocks)
            
            if df is None or df.empty:
                st.error("数据获取失败，请检查配置")
                st.stop()
            
            st.success(f"✅ 数据获取完成: {len(df)} 只股票 ({data_source})")
            progress_bar.progress(30)
            
            # 2. 运行选股
            scored, top_picks = run_screening(df, use_ml=use_ml)
            progress_bar.progress(70)
            
            # 3. 展示结果
            progress_bar.progress(90)
            
            # Top N 表格
            st.markdown(f"#### 🏆 Top {top_n} 推荐股票")
            
            # 准备展示数据
            display_cols = ['code', 'name', 'sw_industry_name', 'composite_score']
            dim_cols = [c for c in top_picks.columns if c.endswith('_score') and c != 'composite_score_raw']
            display_cols.extend(dim_cols)
            if 'industry_rank' in top_picks.columns:
                display_cols.append('industry_rank')
            
            display_df = top_picks[[c for c in display_cols if c in top_picks.columns]].copy()
            
            # 重命名列
            rename_map = {
                'code': '代码',
                'name': '名称',
                'sw_industry_name': '行业',
                'composite_score': '综合得分',
                'value_score': '价值',
                'quality_score': '质量',
                'growth_score': '增长',
                'momentum_score': '动量',
                'expectation_score': '预期差距',
                'industry_rank': '行业排名',
            }
            display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "综合得分": st.column_config.ProgressColumn("综合得分", min_value=0, max_value=100, format="%.1f"),
                    "价值": st.column_config.NumberColumn("价值", format="%.2f"),
                    "质量": st.column_config.NumberColumn("质量", format="%.2f"),
                    "增长": st.column_config.NumberColumn("增长", format="%.2f"),
                    "动量": st.column_config.NumberColumn("动量", format="%.2f"),
                    "预期差距": st.column_config.NumberColumn("预期差距", format="%.2f"),
                }
            )
            
            # 雷达图
            st.markdown("#### 📊 Top 10 股票维度雷达图")
            
            radar_dims = [c for c in ['value_score', 'quality_score', 'growth_score', 'momentum_score', 'expectation_score'] 
                         if c in top_picks.columns]
            if radar_dims:
                radar_df = top_picks.head(10).copy()
                
                fig = go.Figure()
                colors = px.colors.qualitative.Set3[:10]
                
                for idx, (_, row) in enumerate(radar_df.iterrows()):
                    values = [row.get(d, 0) for d in radar_dims]
                    values.append(values[0])  # 闭合
                    
                    categories = [d.replace('_score', '').title() for d in radar_dims]
                    categories.append(categories[0])
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=f"{row['name'][:8]} ({row['code']})",
                        line_color=colors[idx % len(colors)],
                        opacity=0.3,
                    ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[-3, 3])),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                    height=500,
                    title="Top 10 股票 - 五维度得分分布",
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 综合得分分布
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
            
            progress_bar.progress(100)
            st.success("✅ 选股完成！")
            
            # 保存到session state供其他tab使用
            st.session_state['scored'] = scored
            st.session_state['top_picks'] = top_picks
    else:
        st.info("👆 点击上方按钮运行选股模型")
        
        # 展示说明
        st.markdown("""
        #### 选股流程说明
        1. **数据获取**: 从选定市场获取股票财务/价格/预期数据
        2. **因子计算**: 计算19个标准化因子（Value/Quality/Growth/Momentum/Expectation）
        3. **维度评分**: 5维度等权平均 + z-score标准化
        4. **综合排名**: 加权综合得分 → 0-100百分位排名
        5. **行业内排名**: 每个行业内独立排名（GRP风格）
        6. **Top N输出**: 综合得分最高的N只股票
        """)

# ---------------------------------------------------------------------------
# Tab 2: 回测分析
# ---------------------------------------------------------------------------
with tab2:
    st.markdown("### 📈 Walk-forward 回测分析")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        run_bt = st.button("▶️ 运行回测", type="primary", use_container_width=True)
    with col2:
        st.info(f"参数: {rebalance_freq}调仓 | 成本: {transaction_cost*100:.2f}% | Top {top_n}")
    
    if run_bt:
        if not modules.get('backtest', False):
            st.error("回测引擎未加载，无法运行回测")
        else:
            with st.spinner("正在运行回测（约需10-30秒）..."):
                from backtest.engine import BacktestConfig, run_backtest, analyze_backtest_results
                
                config = BacktestConfig(
                    start_date='2022-01-01',
                    end_date='2023-12-31',
                    rebalance_freq=rebalance_freq,
                    top_n=top_n,
                    transaction_cost=transaction_cost,
                )
                
                # 使用mock数据运行回测
                results = run_backtest(config=config)
                
                if results.empty:
                    st.error("回测结果为空")
                else:
                    analysis = analyze_backtest_results(results)
                    
                    # 绩效指标卡片
                    st.markdown("#### 📊 回测绩效摘要")
                    
                    metric_cols = st.columns(4)
                    metrics = [
                        ("总收益", f"{analysis.get('total_return', 0):.2f}%", "#27AE60"),
                        ("年化收益", f"{analysis.get('annualized_return', 0):.2f}%", "#2980B9"),
                        ("最大回撤", f"{analysis.get('max_drawdown', 0):.2f}%", "#E74C3C"),
                        ("夏普比率", f"{analysis.get('sharpe_ratio', 0):.3f}", "#8E44AD"),
                    ]
                    
                    for col, (label, value, color) in zip(metric_cols, metrics):
                        with col:
                            st.markdown(f"""
                            <div style="background-color:{color}15; border-left:4px solid {color}; padding:1rem; border-radius:8px;">
                                <div style="font-size:0.9rem; color:#7F8C8D;">{label}</div>
                                <div style="font-size:1.5rem; font-weight:bold; color:{color};">{value}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    metric_cols2 = st.columns(4)
                    metrics2 = [
                        ("胜率", f"{analysis.get('win_rate', 0):.1f}%", "#F39C12"),
                        ("平均IC", f"{analysis.get('avg_ic', 0):.4f}", "#16A085"),
                        ("ICIR", f"{analysis.get('icir', 0):.3f}", "#D35400"),
                        ("平均换手", f"{analysis.get('avg_turnover', 0):.3f}", "#2C3E50"),
                    ]
                    
                    for col, (label, value, color) in zip(metric_cols2, metrics2):
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
                        height=400,
                        xaxis_title="日期",
                        yaxis_title="累计收益 (%)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 超额收益分布
                    st.markdown("#### 📊 超额收益分布")
                    fig2 = px.histogram(
                        results, x='excess_return', nbins=20,
                        title="每期超额收益分布",
                        labels={'excess_return': '超额收益 (%)', 'count': '期数'},
                        color_discrete_sequence=['#3498DB'],
                    )
                    fig2.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="零线")
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # 详细结果表格
                    with st.expander("📋 查看每期详细结果"):
                        display_results = results[['trade_date', 'period_return', 'benchmark_return', 
                                                     'excess_return', 'cumulative_return', 'turnover', 'ic']].copy()
                        display_results.columns = ['调仓日', '组合收益(%)', '基准收益(%)', 
                                                   '超额收益(%)', '累计收益(%)', '换手率', 'Rank IC']
                        st.dataframe(display_results.round(3), use_container_width=True, hide_index=True)
                    
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

# ---------------------------------------------------------------------------
# Tab 3: 因子分析
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("### 🔬 因子分析")
    
    if 'scored' in st.session_state:
        scored = st.session_state['scored']
        
        # 因子分布
        st.markdown("#### 📊 各维度得分分布")
        
        dim_cols = [c for c in ['value_score', 'quality_score', 'growth_score', 'momentum_score', 'expectation_score']
                   if c in scored.columns]
        
        if dim_cols:
            fig = make_subplots(rows=2, cols=3, subplot_titles=[d.replace('_score', '').title() for d in dim_cols])
            
            positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
            colors = ['#3498DB', '#E74C3C', '#27AE60', '#F39C12', '#9B59B6']
            
            for (dim, pos, color) in zip(dim_cols, positions, colors):
                fig.add_trace(
                    go.Histogram(x=scored[dim], name=dim.replace('_score', '').title(), 
                                marker_color=color, opacity=0.7),
                    row=pos[0], col=pos[1]
                )
            
            fig.update_layout(height=500, showlegend=False, title_text="五维度得分分布")
            st.plotly_chart(fig, use_container_width=True)
        
        # 因子相关性
        st.markdown("#### 🔗 维度得分相关性矩阵")
        if dim_cols:
            corr = scored[dim_cols].corr()
            corr.index = [d.replace('_score', '').title() for d in corr.index]
            corr.columns = [d.replace('_score', '').title() for d in corr.columns]
            
            fig = px.imshow(corr, text_auto='.2f', aspect="auto",
                           color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                           title="维度得分相关性")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("请先运行 🎯 实时选股，获取因子分析数据")

# ---------------------------------------------------------------------------
# Tab 4: 行业分布
# ---------------------------------------------------------------------------
with tab4:
    st.markdown("### 🏭 行业分布分析")
    
    if 'scored' in st.session_state:
        scored = st.session_state['scored']
        
        if 'sw_industry_name' in scored.columns:
            # 行业平均得分
            dim_cols = [c for c in ['value_score', 'quality_score', 'growth_score', 'momentum_score', 'expectation_score']
                       if c in scored.columns]
            
            if dim_cols:
                industry_data = scored.groupby('sw_industry_name')[dim_cols + ['composite_score']].mean().reset_index()
                industry_data = industry_data.sort_values('composite_score', ascending=False)
                
                # 重命名
                rename = {d: d.replace('_score', '').title() for d in dim_cols}
                rename['composite_score'] = '综合得分'
                rename['sw_industry_name'] = '行业'
                industry_data = industry_data.rename(columns=rename)
                
                st.markdown("#### 📊 行业平均得分排名")
                st.dataframe(industry_data.round(3), use_container_width=True, hide_index=True)
                
                # 行业热力图
                st.markdown("#### 🔥 行业维度热力图")
                heatmap_cols = [d.replace('_score', '').title() for d in dim_cols]
                
                fig = px.imshow(
                    industry_data.set_index('行业')[heatmap_cols],
                    text_auto='.2f',
                    aspect="auto",
                    color_continuous_scale='RdYlGn',
                    title="行业 × 维度 平均得分热力图",
                    zmin=-2, zmax=2,
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 行业股票数量
            st.markdown("#### 📊 行业股票数量分布")
            industry_counts = scored['sw_industry_name'].value_counts().reset_index()
            industry_counts.columns = ['行业', '股票数量']
            
            fig = px.bar(industry_counts, x='行业', y='股票数量',
                        title="各行业股票数量", color='股票数量',
                        color_continuous_scale='Blues')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("数据中缺少行业分类信息")
    else:
        st.info("请先运行 🎯 实时选股，获取行业分布数据")

# ---------------------------------------------------------------------------
# Tab 5: 系统状态
# ---------------------------------------------------------------------------
with tab5:
    st.markdown("### ⚙️ 系统状态")
    
    st.markdown("#### 📦 模块加载状态")
    
    module_df = pd.DataFrame([
        {"模块": name, "状态": status[0], "说明": status[1]}
        for name, status in module_status.items()
    ])
    st.dataframe(module_df, use_container_width=True, hide_index=True)
    
    st.markdown("#### 🔧 系统信息")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.metric("Python版本", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    with info_col2:
        st.metric("当前时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    with info_col3:
        st.metric("项目路径", str(project_root))
    
    st.markdown("#### 📋 19因子清单")
    
    try:
        from core.factor_engine import FACTOR_DEFINITIONS, FACTOR_CATEGORIES
        
        factor_df = pd.DataFrame(FACTOR_DEFINITIONS)
        factor_df['category_cn'] = factor_df['category'].map({
            'value': '价值', 'quality': '质量', 'growth': '增长', 
            'momentum': '动量', 'expectation': '预期差距'
        })
        
        display = factor_df[['category_cn', 'name', 'column', 'direction']].copy()
        display.columns = ['类别', '名称', '字段名', '方向']
        display['方向'] = display['方向'].map({1: '正向（越高越好）', -1: '负向（越低越好）'})
        
        st.dataframe(display, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"无法加载因子定义: {e}")
    
    st.markdown("#### ⚖️ 评分权重配置")
    
    try:
        from core.scoring_engine import FACTOR_WEIGHTS
        
        weight_df = pd.DataFrame([
            {"维度": k.title(), "权重": f"{v*100:.0f}%"}
            for k, v in FACTOR_WEIGHTS.items()
        ])
        st.dataframe(weight_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"无法加载权重配置: {e}")
    
    st.markdown("---")
    st.markdown("""
    **Mini-GRP v3.2** | 参考 Principal Global Investors GRP 设计  
    19因子 × 5维度 × ML增强 × Walk-forward回测  
    [GitHub](https://github.com) | [文档](docs/Mini-GRP-v32-Technical-Report.md)
    """)

# ---------------------------------------------------------------------------
# 页脚
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("Mini-GRP v3.2 © 2025 | 跨市场量化选股研究平台 | 数据仅供研究参考，不构成投资建议")
