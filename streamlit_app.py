"""Streamlit application for Mini-GRP v3.4."""

from __future__ import annotations

import hashlib
import os
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics.factor_research import extract_model_importance, run_factor_research
from backtest.engine import BacktestConfig, analyze_backtest_results, run_backtest
from core.main import score_universe
from data.unified_fetcher import DataSourceUnavailable, fetch_a_share_data, fetch_us_data
from data.tushare_client import TushareClientError, probe_tushare_connection
from research.backtest import run_snapshot_backtest
from research.inference import apply_ml_overlay
from research.storage import ResearchStore
from research.trainer import prepare_training_panel
from utils.mock import generate_mock_data


DIMENSION_LABELS = {
    "value_score": "Value",
    "quality_score": "Quality",
    "growth_score": "Growth",
    "momentum_score": "Momentum",
    "expectation_score": "Expectation",
}

LINEAR_MODE = "Linear baseline"
APPROVED_ML_MODE = "Approved ML overlay"
EXPERIMENTAL_ML_MODE = "Experimental ML candidate"


def _factor_label(name: str) -> str:
    return name.removesuffix("_z").replace("_", " ").title()


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


def _secret_value(name: str) -> str:
    """Read a top-level Streamlit secret without requiring a local secrets file."""
    secret_paths = [Path.home() / ".streamlit" / "secrets.toml", Path.cwd() / ".streamlit" / "secrets.toml"]
    if not any(path.exists() for path in secret_paths):
        return ""
    try:
        value = st.secrets[name]
    except Exception:
        return ""
    return str(value).strip()


def _configure_research_api() -> str:
    """Load provider settings from Secrets and optional sidebar overrides."""
    env_token = os.environ.get("TUSHARE_TOKEN", "").strip()
    env_url = os.environ.get("TUSHARE_API_URL", "").strip()
    env_alpha = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()
    current_token = env_token or _secret_value("TUSHARE_TOKEN")
    current_url = env_url or _secret_value("TUSHARE_API_URL")
    current_alpha = env_alpha or _secret_value("ALPHA_VANTAGE_API_KEY")

    with st.sidebar.expander("Research API configuration"):
        token_override = st.text_input(
            "Tushare token override",
            type="password",
            help="Leave blank to use TUSHARE_TOKEN from Streamlit Secrets.",
        ).strip()
        api_url = st.text_input(
            "Tushare API URL",
            value=current_url,
            placeholder="https://ts.gyzcloud.top/api",
            help="Use the proxy URL supplied with the token, or leave blank for official Tushare.",
        ).strip()
        alpha_override = st.text_input(
            "Alpha Vantage key override",
            type="password",
            help="Leave blank to use ALPHA_VANTAGE_API_KEY from Streamlit Secrets.",
        ).strip()

        effective_token = token_override or current_token
        effective_alpha = alpha_override or current_alpha
        if effective_token:
            os.environ["TUSHARE_TOKEN"] = effective_token
        if api_url:
            os.environ["TUSHARE_API_URL"] = api_url
        else:
            os.environ.pop("TUSHARE_API_URL", None)
        if effective_alpha:
            os.environ["ALPHA_VANTAGE_API_KEY"] = effective_alpha

        status = "configured" if effective_token else "missing"
        st.caption(f"Tushare token: {status}. Endpoint: {api_url or 'official API'}")
        if st.button("Test Tushare connection", use_container_width=True):
            try:
                summary = probe_tushare_connection(effective_token, api_url)
                st.success(
                    f"Connected: {summary['listed_stocks']:,} listed stocks; "
                    f"latest open date {summary['trade_date']}."
                )
            except TushareClientError as exc:
                st.error(str(exc))

    signature_value = f"{effective_token}|{api_url}|{effective_alpha}"
    return hashlib.sha256(signature_value.encode("utf-8")).hexdigest()[:12]


def _database_url() -> str | None:
    value = os.environ.get("DATABASE_URL", "").strip() or _secret_value("DATABASE_URL")
    return value or None


@st.cache_data(ttl=3600, show_spinner=False)
def _load_universe(
    data_mode: str,
    market: str,
    n_stocks: int,
    seed: int,
    provider_signature: str,
) -> pd.DataFrame:
    del provider_signature
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


def _screening_tab(
    data_mode: str,
    market: str,
    n_stocks: int,
    top_n: int,
    seed: int,
    provider_signature: str,
    scoring_mode: str,
    database_url: str | None,
) -> None:
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Stock screening")
        st.write("Build a cross-sectional ranking from 19 candidate factors across five research dimensions.")
    with right:
        run = st.button("Run screening", type="primary", use_container_width=True)

    if run:
        try:
            model = None
            if scoring_mode in {APPROVED_ML_MODE, EXPERIMENTAL_ML_MODE}:
                store = ResearchStore(database_url)
                model_status = "approved" if scoring_mode == APPROVED_ML_MODE else "candidate"
                model = store.latest_model(model_status)
                if model is None:
                    st.error(f"No {model_status} ML model is available in the model registry.")
                    st.caption("Open Model registry to review the candidate validation metrics.")
                    return
            with st.spinner("Loading and scoring the universe..."):
                universe = _load_universe(data_mode, market, n_stocks, seed, provider_signature)
                scored, baseline_top = score_universe(universe, top_n)
                model_version = "linear-v3.4"
                model_summary = None
                if model is not None:
                    scored = apply_ml_overlay(
                        scored,
                        model_record=model,
                        allow_candidate=scoring_mode == EXPERIMENTAL_ML_MODE,
                    )
                    top_picks = scored.head(top_n).copy()
                    model_version = str(top_picks["model_version"].iloc[0])
                    selected_name = model.metrics.get("selected_model", "unknown")
                    validation = model.metrics.get("validation", {}).get(selected_name, {})
                    model_summary = {
                        "algorithm": selected_name,
                        "mean_rank_ic": validation.get("mean_rank_ic"),
                        "mean_spread": validation.get("mean_top_bottom_spread"),
                        "overlay_weight": float(model.bundle.get("overlay_weight", 0.15)),
                    }
                else:
                    top_picks = baseline_top
                st.session_state["screening"] = (
                    scored,
                    top_picks,
                    scoring_mode,
                    model_version,
                    model_summary,
                )
        except DataSourceUnavailable as exc:
            st.error(str(exc))
        except Exception as exc:
            st.exception(exc)

    if "screening" not in st.session_state:
        st.info("Run the model to generate the ranked universe.")
        return

    scored, top_picks, applied_mode, model_version, model_summary = st.session_state["screening"]
    source = str(scored.get("data_source", pd.Series(["unknown"])).iloc[0])
    coverage = float(scored.get("factor_coverage", pd.Series([1.0])).mean())
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Universe", f"{len(scored):,}")
    c2.metric("Selected", len(top_picks))
    c3.metric("Data source", source)
    c4.metric("Mean factor coverage", f"{coverage:.0%}")
    model_label = {
        APPROVED_ML_MODE: "Approved ML",
        EXPERIMENTAL_ML_MODE: "Experimental ML",
    }.get(applied_mode, "Linear")
    c5.metric("Scoring model", model_label)

    if applied_mode == APPROVED_ML_MODE:
        st.caption(f"Approved model: {model_version}. ML contribution is capped at 30% of the final score.")
    elif applied_mode == EXPERIMENTAL_ML_MODE:
        st.warning(
            "Experimental ML is active. This candidate failed the production approval thresholds, "
            "so its ranking is for research comparison rather than investment use."
        )

    if model_summary is not None:
        m1, m2, m3, m4 = st.columns(4)
        rank_ic = model_summary["mean_rank_ic"]
        spread = model_summary["mean_spread"]
        m1.metric("ML algorithm", str(model_summary["algorithm"]).replace("_", " ").title())
        m2.metric("Validation rank IC", "N/A" if rank_ic is None else f"{rank_ic:.4f}")
        m3.metric("Top-bottom spread", "N/A" if spread is None else f"{spread:.2%}")
        m4.metric("ML weight", f"{model_summary['overlay_weight']:.0%}")

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


def _backtest_tab(top_n: int, seed: int, database_url: str | None) -> None:
    st.subheader("Point-in-time backtest")
    dataset = st.radio(
        "Dataset",
        ["Stored research snapshots", "Synthetic pipeline check"],
        horizontal=True,
    )
    if dataset == "Stored research snapshots":
        st.info(
            "Uses factors stored at each month-end and the following 20-trading-day return. "
            "The current research backtest covers the interpretable linear baseline; ML evidence is shown in Model registry."
        )
        cost_pct = st.slider("One-way transaction cost (%)", 0.0, 0.5, 0.1, 0.05, key="research_cost")
        if st.button("Run research backtest", use_container_width=True):
            try:
                with st.spinner("Scoring stored month-end snapshots..."):
                    results, metrics = run_snapshot_backtest(
                        ResearchStore(database_url),
                        top_n=top_n,
                        transaction_cost=cost_pct / 100.0,
                    )
                    st.session_state["research_backtest"] = (results, metrics)
            except Exception as exc:
                st.error(str(exc))
        if "research_backtest" not in st.session_state:
            return
        results, metrics = st.session_state["research_backtest"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Annualized return", f"{metrics['annualized_return']:.2%}")
        c2.metric("Annualized excess", f"{metrics['annualized_excess']:.2%}")
        c3.metric("Max drawdown", f"{metrics['max_drawdown']:.2%}")
        c4.metric("Average turnover", f"{metrics['average_turnover']:.1%}")
        curve = results[["snapshot_date", "portfolio_nav", "benchmark_nav"]].copy()
        curve["snapshot_date"] = pd.to_datetime(curve["snapshot_date"])
        curve = curve.melt("snapshot_date", var_name="Series", value_name="NAV")
        st.plotly_chart(
            px.line(curve, x="snapshot_date", y="NAV", color="Series", title="Stored-snapshot equity curve"),
            use_container_width=True,
        )
        st.dataframe(results.drop(columns=["selected_codes"], errors="ignore"), use_container_width=True, hide_index=True)
        return

    st.warning(
        "This deterministic synthetic dataset validates the pipeline and accounting only; it is not investment evidence."
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


def _model_registry_tab(database_url: str | None) -> None:
    st.subheader("Research data and model registry")
    try:
        store = ResearchStore(database_url)
        status = store.status()
        model = store.latest_model("approved")
        candidate = store.latest_model("candidate")
    except Exception as exc:
        st.error(f"Research database unavailable: {exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Snapshot dates", status["snapshot_dates"])
    c2.metric("Snapshot rows", f"{status['snapshot_rows']:,}")
    c3.metric("Models trained", status["model_count"])
    c4.metric("Latest snapshot", status["snapshot_end"] or "None")

    if model is None:
        st.warning("No approved model is available. Production screening remains on the linear baseline.")
    else:
        st.success(f"Approved production model: {model.version}")

    displayed_model = model or candidate
    if displayed_model is None:
        st.info("No trained model is available yet.")
        return

    if model is None and candidate is not None:
        st.info(
            "The latest candidate can now be selected as Experimental ML in the sidebar. "
            "It remains clearly separated from approved production inference."
        )

    metrics = displayed_model.metrics
    selected = metrics.get("validation", {}).get(metrics.get("selected_model", ""), {})
    status_label = "Approved" if displayed_model.status == "approved" else "Experimental candidate"
    st.subheader(status_label)
    st.caption(
        f"{displayed_model.version} | {displayed_model.trained_from} to {displayed_model.trained_through} | "
        f"{metrics.get('training_rows', 0):,} training rows"
    )
    m1, m2, m3 = st.columns(3)
    mean_ic = selected.get("mean_rank_ic")
    spread = selected.get("mean_top_bottom_spread")
    m1.metric("Walk-forward rank IC", "N/A" if mean_ic is None else f"{mean_ic:.3f}")
    m2.metric("Top-bottom forward spread", "N/A" if spread is None else f"{spread:.2%}")
    m3.metric("Selected algorithm", metrics.get("selected_model", "unknown"))

    failed_checks = []
    if mean_ic is None or mean_ic <= 0:
        failed_checks.append("rank IC must be positive")
    if spread is None or spread <= 0:
        failed_checks.append("top-bottom spread must be positive")
    if displayed_model.status != "approved" and failed_checks:
        st.warning("Approval checks not passed: " + "; ".join(failed_checks) + ".")

    folds = pd.DataFrame(selected.get("folds", []))
    if not folds.empty:
        folds["date"] = pd.to_datetime(folds["date"])
        st.plotly_chart(
            px.bar(folds, x="date", y="rank_ic", title="Out-of-sample rank IC by validation month"),
            use_container_width=True,
        )
        st.dataframe(folds, use_container_width=True, hide_index=True)

    with st.expander("Model features and factor diagnostics"):
        st.write(", ".join(displayed_model.features))
        st.dataframe(pd.DataFrame(metrics.get("factor_metrics", [])), use_container_width=True, hide_index=True)

    importance = extract_model_importance(displayed_model.bundle)
    if not importance.empty:
        importance["label"] = importance["feature"].map(_factor_label)
        st.plotly_chart(
            px.bar(
                importance.head(15).sort_values("importance"),
                x="importance",
                y="label",
                orientation="h",
                title="Native model feature importance",
                labels={"importance": "Normalized importance", "label": "Feature"},
            ),
            use_container_width=True,
        )
        st.caption(
            "Gradient Boosting uses normalized split importance; Ridge uses absolute coefficient magnitude. "
            "This is a model diagnostic, not a SHAP attribution."
        )


def _factor_research_tab(database_url: str | None) -> None:
    st.subheader("Single-factor research")
    st.write(
        "Evaluate every usable feature on stored point-in-time snapshots before it enters the combined score or ML model."
    )
    st.caption(
        "Pass rule: mean Rank IC > 0.03, ICIR > 0.5, positive-IC ratio > 55%, and positive Q5-Q1 spread."
    )

    if st.button("Run factor diagnostics", type="primary", use_container_width=True):
        try:
            with st.spinner("Rebuilding dated factor panels and running cross-sectional tests..."):
                snapshots = ResearchStore(database_url).load_snapshots(labelled_only=True)
                panel = prepare_training_panel(snapshots)
                features = list(panel.attrs["model_features"])
                result = run_factor_research(panel, features)
                st.session_state["factor_research"] = result
        except Exception as exc:
            st.exception(exc)

    if "factor_research" not in st.session_state:
        st.info("Run diagnostics to calculate IC, quintile, turnover, correlation, and decay results.")
        return

    result = st.session_state["factor_research"]
    summary = result.summary.copy()
    passing = int(summary["passed"].sum())
    best = summary.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Factors tested", len(summary))
    c2.metric("Passing factors", passing)
    c3.metric("Best mean Rank IC", f"{best['mean_rank_ic']:.3f}")
    c4.metric("Best factor", _factor_label(str(best["factor"])))

    chart = summary.copy()
    chart["label"] = chart["factor"].map(_factor_label)
    chart["status"] = chart["passed"].map({True: "Pass", False: "Fail"})
    st.plotly_chart(
        px.bar(
            chart.sort_values("mean_rank_ic"),
            x="mean_rank_ic",
            y="label",
            color="status",
            orientation="h",
            title="Mean monthly Rank IC by factor",
            color_discrete_map={"Pass": "#0f766e", "Fail": "#94a3b8"},
            labels={"mean_rank_ic": "Mean Rank IC", "label": "Factor"},
        ),
        use_container_width=True,
    )

    display = summary.copy()
    display["factor"] = display["factor"].map(_factor_label)
    percent_columns = [
        "coverage",
        "ic_positive_ratio",
        "mean_top_bottom_spread",
        "long_short_annualized",
        "factor_turnover",
    ]
    for column in percent_columns:
        display[column] = display[column].map(lambda value: "N/A" if pd.isna(value) else f"{value:.2%}")
    for column in ["mean_rank_ic", "ic_std", "icir", "long_short_sharpe", "quantile_monotonicity", "rank_autocorrelation"]:
        display[column] = display[column].map(lambda value: "N/A" if pd.isna(value) else f"{value:.3f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    selected = st.selectbox(
        "Factor detail",
        summary["factor"].tolist(),
        format_func=_factor_label,
    )
    detail_left, detail_right = st.columns(2)
    ic = result.ic_series[result.ic_series["factor"] == selected].copy()
    if not ic.empty:
        ic["snapshot_date"] = pd.to_datetime(ic["snapshot_date"])
        with detail_left:
            st.plotly_chart(
                px.line(ic, x="snapshot_date", y="rank_ic", markers=True, title="Monthly Rank IC"),
                use_container_width=True,
            )

    quantiles = result.quantile_returns[result.quantile_returns["factor"] == selected]
    if not quantiles.empty:
        mean_quantiles = quantiles.groupby("quantile", as_index=False)["forward_return"].mean()
        mean_quantiles["quantile"] = mean_quantiles["quantile"].map(lambda value: f"Q{value}")
        with detail_right:
            st.plotly_chart(
                px.bar(
                    mean_quantiles,
                    x="quantile",
                    y="forward_return",
                    title="Average forward return by quintile",
                    labels={"forward_return": "Mean 20-day return", "quantile": "Portfolio"},
                ),
                use_container_width=True,
            )

    decay = result.decay[result.decay["factor"] == selected]
    st.plotly_chart(
        px.line(
            decay,
            x="lag_months",
            y="mean_rank_ic",
            markers=True,
            title="IC lead/decay profile",
            labels={"lag_months": "Return window lead (months)", "mean_rank_ic": "Mean Rank IC"},
        ),
        use_container_width=True,
    )
    st.caption(
        "Lag 0 pairs the factor with its next 20-trading-day return. Lag N pairs today's factor with the "
        "20-day return window beginning N monthly snapshots later."
    )

    correlation = result.factor_correlation.copy()
    correlation.index = [_factor_label(value) for value in correlation.index]
    correlation.columns = [_factor_label(value) for value in correlation.columns]
    st.plotly_chart(
        px.imshow(
            correlation,
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            title="Average cross-sectional factor rank correlation",
            aspect="auto",
        ),
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title="Mini-GRP v3.4", layout="wide", initial_sidebar_state="expanded")
    _style()
    st.markdown(
        "<div class='hero'><h1>Mini-GRP v3.4</h1>"
        "<p>Point-in-time factor research, governed machine learning, and production screening.</p></div>",
        unsafe_allow_html=True,
    )

    st.sidebar.header("Run configuration")
    data_mode = st.sidebar.radio("Data mode", ["Synthetic demo", "Research data"])
    market = st.sidebar.selectbox("Market", ["CN", "US"], disabled=data_mode == "Synthetic demo")
    n_stocks = st.sidebar.slider("Universe size", 50, 500, 200, 50)
    top_n = st.sidebar.slider("Top N", 5, 50, 20, 5)
    scoring_mode = st.sidebar.selectbox(
        "Scoring model",
        [LINEAR_MODE, EXPERIMENTAL_ML_MODE, APPROVED_ML_MODE],
        help="Experimental ML runs the latest candidate even when it has not passed production approval.",
    )
    seed = st.sidebar.number_input("Demo seed", min_value=1, max_value=999999, value=42)

    provider_signature = _configure_research_api()
    database_url = _database_url()

    css_class = "mode-demo" if data_mode == "Synthetic demo" else "mode-research"
    st.markdown(
        f"<span class='{css_class}'>{data_mode}</span>",
        unsafe_allow_html=True,
    )
    screen_tab, factor_tab, model_tab, backtest_tab, method_tab = st.tabs(
        ["Screening", "Factor research", "Model registry", "Backtest", "Methodology"]
    )
    with screen_tab:
        _screening_tab(
            data_mode,
            market,
            n_stocks,
            top_n,
            int(seed),
            provider_signature,
            scoring_mode,
            database_url,
        )
    with factor_tab:
        _factor_research_tab(database_url)
    with model_tab:
        _model_registry_tab(database_url)
    with backtest_tab:
        _backtest_tab(top_n, int(seed), database_url)
    with method_tab:
        st.subheader("Research process")
        st.markdown(
            """
            1. Acquire a point-in-time universe and retain data provenance.
            2. Validate factor availability; unusable or constant factors are excluded.
            3. Winsorize and standardize factors within each market, then apply directionality.
            4. Build Value, Quality, Growth, Momentum, and Expectation dimension scores.
            5. Store month-end point-in-time snapshots and calculate the following 20-trading-day label.
            6. Validate each factor with monthly Rank IC, true ICIR, quintile monotonicity, turnover, correlation, and decay tests.
            7. Validate Ridge and Gradient Boosting with expanding-window, out-of-sample folds.
            8. Register a model only when rank IC and top-minus-bottom spread are both positive.
            9. Blend an approved ML score into the linear baseline with a hard 30% maximum weight.
               A separately labelled experimental mode can run the latest candidate for comparison.
            10. Backtest stored snapshots with explicit turnover costs and an equal-weight universe benchmark.

            The synthetic mode is for demonstrations and regression tests. Research mode refuses silent fallback to synthetic data.
            """
        )


if __name__ == "__main__":
    main()
