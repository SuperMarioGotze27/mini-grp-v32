# Mini-GRP v3.3

An auditable multi-factor stock screening and walk-forward research framework inspired by publicly described ideas behind Principal's Global Research Platform (GRP).

Mini-GRP is a research and interview project, not an automated trading product. It turns a broad stock universe into an explainable shortlist, while keeping synthetic demonstrations visibly separated from genuine market data.

## What Changed in v3.3

- Restored the complete modular codebase to GitHub instead of publishing only deployment stubs.
- Fixed the broken `app.py` entry point and removed duplicated scoring/backtest logic.
- Added explicit `demo` and `research` data modes. Research mode refuses synthetic fallback.
- Added provenance fields: `data_source`, `is_mock`, `as_of_date`, `factor_coverage`, and `expectation_source`.
- Standardized factors within each market before cross-market ranking.
- Excluded missing or constant factors and dynamically re-normalized active dimension weights.
- Rebuilt the walk-forward engine with deterministic dates, turnover-based costs, equal-weight benchmark, frequency-aware annualization, IC/ICIR, and hard failure gates.
- Added reproducible CSV/JSON artifacts, automated tests, GitHub Actions CI, Docker support, and a verified Streamlit interface.

## Research Workflow

```text
Market data / synthetic demo
        |
        v
Provenance and schema validation
        |
        v
Winsorization -> market-local z-score -> factor direction
        |
        v
Value / Quality / Growth / Momentum / Expectation
        |
        v
Dynamic weighted score -> percentile rank -> industry rank
        |
        +--------------------+
        |                    |
        v                    v
Top-N screening       Walk-forward evaluation
```

## Factor Model

| Dimension | Weight | Factors |
|---|---:|---|
| Value | 25% | PE, PB, PS, EV/EBITDA, dividend yield |
| Quality | 25% | ROE, ROA, gross margin, net margin, debt/equity |
| Growth | 15% | revenue growth, profit growth, FCF yield |
| Momentum | 15% | 1-month, 3-month, 12-month return |
| Expectation | 20% | SUE, EPS revision, rating revision |

If a dimension has no usable provider data, it is excluded and the remaining weights are normalized. Real-data adapters no longer fabricate expectation factors.

## Quick Start

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

Run deterministic demo screening:

```bash
python -m core.main --mode screen --data-mode demo --max-stocks 200 --top-n 20
```

Run the synthetic walk-forward pipeline check:

```bash
python -m core.main --mode backtest --data-mode demo --start-date 2022-01-01 --end-date 2024-12-31
```

## Research Data Mode

Install optional provider adapters:

```bash
pip install -r requirements-research.txt
```

Configure one or more credentials:

```bash
set TUSHARE_TOKEN=your_token
set ALPHA_VANTAGE_API_KEY=your_key
```

Then run, for example:

```bash
python -m core.main --mode screen --data-mode research --market cn --max-stocks 100
```

Research mode raises a clear error when genuine data cannot be obtained. It never substitutes demo data.

## Outputs

Screening writes:

- `screening_universe.csv`
- `top_picks.csv`
- `screening_manifest.json`

Backtesting writes:

- `periods.csv`
- `equity.csv`
- `holdings.csv`
- `metrics.json`
- `run_manifest.json`
- `equity_curve.png`

Each backtest period must contain selected holdings, valid returns, and finite NAV values. Invalid runs fail instead of returning zero-filled performance.

## Validation

```bash
pip install -r requirements-dev.txt
python -m compileall -q .
pytest
```

The CI workflow repeats compilation, tests, demo screening, and demo backtesting on every push and pull request.

## Project Structure

```text
core/              factor processing, scoring, CLI orchestration
data/              provider adapters, unified schema, local cache
backtest/          walk-forward engine and performance artifacts
analytics/         experimental analysis utilities
ml/                optional ML overlays and validators
viz/               legacy static reporting utilities
utils/              deterministic synthetic data
tests/              regression and integration tests
docs/               project reports and deployment guidance
streamlit_app.py    canonical web application
app.py              compatibility wrapper
```

## Current Boundaries

- The bundled backtest demo uses synthetic point-in-time data and is not evidence of an investable strategy.
- A production research backtest still requires licensed historical point-in-time fundamentals, survivorship-bias-free constituents, corporate-action-adjusted prices, and release-date-aware analyst expectations.
- Provider field coverage varies. `factor_coverage` and `expectation_source` should be reviewed before interpreting rankings.
- Optional ML modules remain experimental and are deliberately outside the default scoring path.

## Documentation

- [Complete project report](docs/Mini-GRP-v33-Complete-Project-Report.md)
- [Deployment guide](DEPLOY.md)

## Disclaimer

For research, education, and interview demonstration only. Nothing in this repository constitutes investment advice or a representation of historical or future performance.
