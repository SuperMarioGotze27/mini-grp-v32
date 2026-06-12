"""Streamlit application for Mini-GRP v3.3."""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backtest.engine import BacktestConfig, analyze_backtest_results, run_backtest
from core.main import score_universe
from data.unified_fetcher import DataSourceUnavailable, fetch_a_share_data, fetch_us_data
from utils.mock import generate_mock_data


DIMENSION_LABELS = {
    "value_score": "Value",
    "quality_score": "Quality",
    "growth_score": "Growth",
    "momentum_score": "Momentum",
    "expectation_score": "Expectation",
}


def _style() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1220px; padding-top: 2.2rem;}
        .hero {padding: 1.5rem 1.7rem; border: 1px solid #d9e2ec; border-radius: 16px;
               background: linear-gradient(135deg, #f7fafc 0%, #eef6f8 100%); margin-bottom: 1.2rem;}
        .hero h1 {margin: 0; color: #102a43; font-size: 2.2rem;}
        .hero p {margin: .45rem 0 0; color: #486581;}
        .mode-demo {display:inline-block; padding:.25rem .65rem; border-radius:999px;
                    color:#7c2d12; background:#ffedd5; font-size:.82rem; font-weight:700;}
        .mode-research {display:inline-block; padding:.25rem .65rem; border-radius:999px;
                        color:#14532d; background:#dcfce7; font-size:.82rem; font-weight:700;}
        div[data-testid="stMetric"] {border:1px solid #e5e7eb; padding:.9rem; border-radius:12px; background:white;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _load_universe(data_mode: str, market: str, n_stocks: int, seed: int) -> pd.DataFrame:
    if data_mode == "Synthetic demo":
        return generate_mock_data(n_stocks=n_stocks, seed=seed)
    if market == "CN":
        return fetch_a_share_data(max_stocks=n_stocks, allow_mock=False, use_cache=True)
    return fetch_us_data(max_stocks=n_stocks, allow_mock=False, use_cache=True)


def _radar(top_picks: pd.DataFrame) -> go.Figure:
    top = top_picks.iloc[0]
    columns = [column for column in DIMENSION_LABELS if column in top_picks]
    values = [float(top[column]) for column in columns]
    labels = [DIMENSION_LABELS[column] for column in columns]
    fig = go.Figure(
        go.Scatterpolar(r=values + values[:1], theta=labels + labels[:1], fill="toself", line_color="#0f766e")
    )
    fig.update_layout(
        title=f"Top candidate profile: {top['code']}",
        polar=dict(radialaxis=dict(visible=True)),
        height=390,
        margin=dict(l=35, r=35, t=65, b=25),
        showlegend=False,
    )
    return fig


def _screening_tab(data_mode: str, market: str, n_stocks: int, top_n: int, seed: int) -> None:
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Stock screening")
        st.write("Build a cross-sectional ranking from 19 candidate factors across five research dimensions.")
    with right:
        run = st.button("Run screening", type="primary", use_container_width=True)

    if run:
        try:
            with st.spinner("Loading and scoring the universe..."):
                universe = _load_universe(data_mode, market, n_stocks, seed)
                scored, top_picks = score_universe(universe, top_n)
                st.session_state["screening"] = (scored, top_picks)
        except DataSourceUnavailable as exc:
            st.error(str(exc))
        except Exception as exc:
            st.exception(exc)

    if "screening" not in st.session_state:
        st.info("Run the model to generate the ranked universe.")
        return

    scored, top_picks = st.session_state["screening"]
    source = str(scored.get("data_source", pd.Series(["unknown"])).iloc[0])
    coverage = float(scored.get("factor_coverage", pd.Series([1.0])).mean())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Universe", f"{len(scored):,}")
    c2.metric("Selected", len(top_picks))
    c3.metric("Data source", source)
    c4.metric("Mean factor coverage", f"{coverage:.0%}")

    st.dataframe(top_picks, use_container_width=True, hide_index=True)
    st.download_button(
        "Download top picks CSV",
        data=top_picks.to_csv(index=False).encode("utf-8-sig"),
        file_name="mini_grp_top_picks.csv",
        mime="text/csv",
    )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(_radar(top_picks), use_container_width=True)
    with chart_right:
        industry = top_picks["sw_industry_name"].value_counts().reset_index()
        industry.columns = ["Industry", "Count"]
        fig = px.bar(industry, x="Count", y="Industry", orientation="h", title="Top-pick industry distribution")
        fig.update_layout(height=390, margin=dict(l=20, r=20, t=65, b=25), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def _backtest_tab(top_n: int, seed: int) -> None:
    st.subheader("Walk-forward pipeline check")
    st.warning(
        "This page uses deterministic synthetic point-in-time data. It validates the pipeline and accounting, "
        "but it is not investment evidence."
    )
    c1, c2, c3 = st.columns(3)
    start = c1.date_input("Start", date(2022, 1, 1))
    end = c2.date_input("End", date(2024, 12, 31))
    frequency = c3.selectbox("Rebalance", ["monthly", "quarterly"])
    cost_pct = st.slider("One-way transaction cost (%)", 0.0, 0.5, 0.1, 0.05)

    if st.button("Run synthetic backtest", use_container_width=True):
        try:
            config = BacktestConfig(
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                rebalance_freq=frequency,
                top_n=top_n,
                transaction_cost=cost_pct / 100.0,
            )
            with st.spinner("Running walk-forward periods..."):
                results = run_backtest(config=config, demo_seed=seed)
                metrics = analyze_backtest_results(results)
                st.session_state["backtest"] = (results, metrics)
        except Exception as exc:
            st.exception(exc)

    if "backtest" not in st.session_state:
        return
    results, metrics = st.session_state["backtest"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annualized return", f"{metrics['annualized_return']:.2f}%")
    c2.metric("Annualized excess", f"{metrics['annualized_excess']:.2f}%")
    c3.metric("Max drawdown", f"{metrics['max_drawdown']:.2f}%")
    c4.metric("Average IC", "N/A" if metrics["avg_ic"] is None else f"{metrics['avg_ic']:.3f}")
    curve = results[["trade_date", "portfolio_nav", "benchmark_nav"]].copy()
    curve["trade_date"] = pd.to_datetime(curve["trade_date"])
    curve = curve.melt("trade_date", var_name="Series", value_name="NAV")
    fig = px.line(curve, x="trade_date", y="NAV", color="Series", title="Synthetic demo equity curve")
    fig.update_layout(height=430)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(results.drop(columns=["selected_stocks"], errors="ignore"), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Mini-GRP v3.3", layout="wide", initial_sidebar_state="expanded")
    _style()
    st.markdown(
        "<div class='hero'><h1>Mini-GRP v3.3</h1>"
        "<p>Auditable multi-factor stock screening and walk-forward research framework.</p></div>",
        unsafe_allow_html=True,
    )

    st.sidebar.header("Run configuration")
    data_mode = st.sidebar.radio("Data mode", ["Synthetic demo", "Research data"])
    market = st.sidebar.selectbox("Market", ["CN", "US"], disabled=data_mode == "Synthetic demo")
    n_stocks = st.sidebar.slider("Universe size", 50, 500, 200, 50)
    top_n = st.sidebar.slider("Top N", 5, 50, 20, 5)
    seed = st.sidebar.number_input("Demo seed", min_value=1, max_value=999999, value=42)

    with st.sidebar.expander("Research API configuration"):
        tushare = st.text_input("Tushare token", type="password")
        tushare_url = st.text_input(
            "Tushare API URL (optional)",
            placeholder="https://ts.gyzcloud.top/api",
            help="Leave empty for official Tushare API. Fill in for third-party proxy services.",
        )
        alpha = st.text_input("Alpha Vantage key", type="password")
        if tushare:
            os.environ["TUSHARE_TOKEN"] = tushare
        if tushare_url:
            os.environ["TUSHARE_API_URL"] = tushare_url
        if alpha:
            os.environ["ALPHA_VANTAGE_API_KEY"] = alpha

    css_class = "mode-demo" if data_mode == "Synthetic demo" else "mode-research"
    st.markdown(
        f"<span class='{css_class}'>{data_mode}</span>",
        unsafe_allow_html=True,
    )
    screen_tab, backtest_tab, method_tab = st.tabs(["Screening", "Backtest", "Methodology"])
    with screen_tab:
        _screening_tab(data_mode, market, n_stocks, top_n, int(seed))
    with backtest_tab:
        _backtest_tab(top_n, int(seed))
    with method_tab:
        st.subheader("Research process")
        st.markdown(
            """
            1. Acquire a point-in-time universe and retain data provenance.
            2. Validate factor availability; unusable or constant factors are excluded.
            3. Winsorize and standardize factors within each market, then apply directionality.
            4. Build Value, Quality, Growth, Momentum, and Expectation dimension scores.
            5. Combine active dimensions with normalized weights and rank within industry.
            6. For backtests, rebalance only on explicit dates, charge turnover-based costs, and compare with an equal-weight universe benchmark.

            The synthetic mode is for demonstrations and regression tests. Research mode refuses silent fallback to synthetic data.
            """
        )


if __name__ == "__main__":
    main()
