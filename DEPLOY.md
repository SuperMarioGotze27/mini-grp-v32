# Mini-GRP v3.3 Deployment Guide

## Local Verification

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
python -m compileall -q .
pytest
streamlit run streamlit_app.py
```

Open `http://localhost:8501`, run screening, then run the synthetic backtest. The page must identify the current mode as `Synthetic demo` or `Research data`.

## Streamlit Community Cloud

1. Push the repository to GitHub.
2. In Streamlit Community Cloud, create an app from the repository.
3. Set the main file to `streamlit_app.py`.
4. Use `requirements.txt` for the default demo deployment.
5. Add provider secrets only when research data is required:

```toml
TUSHARE_TOKEN = "..."
TUSHARE_API_URL = "https://ts.gyzcloud.top/api"
ALPHA_VANTAGE_API_KEY = "..."
```

The application reads these values directly from Streamlit Secrets. It also accepts temporary sidebar overrides and provides a **Test Tushare connection** button. Do not commit tokens or `.streamlit/secrets.toml`.

## Docker

```bash
docker build -t mini-grp-v33 .
docker run --rm -p 8501:8501 mini-grp-v33
```

For provider adapters, build an image that installs `requirements-research.txt`, then pass credentials as environment variables.

## Deployment Checks

- `python -m compileall -q .` succeeds.
- `pytest` succeeds.
- `streamlit_app.py` imports canonical modules instead of duplicating model logic.
- Synthetic mode is visibly labelled.
- Research mode does not silently fall back to synthetic data.
- API keys and cache files are not tracked.
- GitHub Actions is green.
- The generated manifest records data mode and source.

## Troubleshooting

**Research data unavailable**

Check `TUSHARE_TOKEN` and `TUSHARE_API_URL` in Streamlit Secrets, then use **Test Tushare connection** in the sidebar. Compatible proxy tokens require their supplied API URL; sending them to the official endpoint will fail. Research mode intentionally refuses to disguise missing data with a demo universe.

**Tushare connects but screening raises a pandas TypeError**

Upgrade to the current branch. Tushare can return both `pe`/`pe_ttm` and `ps`/`ps_ttm`; older builds mapped these into duplicate columns and the factor engine received a two-dimensional value.

**Streamlit starts but charts do not appear**

Confirm `plotly` is installed from `requirements.txt`, reload the app, and inspect the Streamlit logs for an upstream scoring exception.

**Pandas frequency error**

v3.3 uses pandas offset objects rather than version-specific `ME`/`QE` aliases. Upgrade to the current branch and reinstall dependencies.

**Backtest produces no output**

The engine now treats zero holdings, missing price windows, NaN NAV, and empty results as hard failures. Inspect the exception instead of relying on zero-filled metrics.
