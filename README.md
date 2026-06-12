# Mini-GRP v3.4

Mini-GRP is an auditable multi-factor stock research system inspired by publicly described ideas behind Principal's Global Research Platform. It turns an equity universe into an explainable shortlist, stores point-in-time monthly snapshots, validates a bounded machine-learning overlay, and exposes the result through Streamlit.

This is a research and interview project, not an automated trading product or investment recommendation.

## v3.4 at a glance

- Live A-share screening through Tushare, including compatible proxy endpoints.
- Eleven currently obtainable A-share factors across Value, Quality, Growth, and Momentum.
- SQL-backed monthly point-in-time snapshots with 20-trading-day forward labels.
- Expanding-window validation of Ridge and Gradient Boosting candidates.
- Model registry with explicit `candidate` and `approved` states.
- Approved ML overlay capped at 30% of the final score; the linear model remains the anchor.
- Real stored-snapshot baseline backtest with turnover costs and an equal-weight benchmark.
- Streamlit pages for screening, model monitoring, research backtesting, and methodology.
- Google Cloud Run service + Cloud Run Job + Cloud SQL deployment pattern.

## Architecture

```text
Tushare live API
      |
      +--> Live cross-section --> factor engine --> linear score --------+
      |                                                                |
      +--> Month-end collector --> SQL snapshots --> walk-forward ML ---+--> bounded final score
                                         |                              |
                                         +--> real snapshot backtest    +--> Streamlit / CSV

Cloud Run service: Streamlit UI and inference
Cloud Run job: incremental collection and model training
Cloud SQL: snapshots and model registry
Secret Manager: provider token and database URL
```

## Factor model

| Dimension | Policy weight | Live historical factors |
|---|---:|---|
| Value | 25% | PE TTM, PB LF, PS TTM, dividend yield |
| Quality | 25% | gross margin, net margin |
| Growth | 15% | revenue YoY, profit YoY |
| Momentum | 15% | 1-month, 3-month, 12-month return |
| Expectation | 20% | reserved for licensed analyst-estimate data |

Unavailable dimensions are excluded and active weights are normalized. Research mode never fabricates missing expectation data.

## Quick start

```powershell
pip install -r requirements-dev.txt
pytest
streamlit run streamlit_app.py
```

Configure real data without committing secrets:

```powershell
$env:TUSHARE_TOKEN="your-token"
$env:TUSHARE_API_URL="https://ts.gyzcloud.top/api"
```

Build monthly history and train the governed overlay:

```powershell
python -m research.cli collect --months 60 --max-stocks 1500
python -m research.cli train --overlay-weight 0.15
python -m research.cli status
python -m research.cli backtest --top-n 20 --transaction-cost 0.001
```

The collector is incremental: completed history is skipped, while the latest two month-ends are refreshed so forward labels can mature.

## Model governance

The ML target is each stock's cross-sectional percentile rank of its future 20-trading-day return. Validation uses earlier dates for training and a later month for testing. A model is registered as `approved` only when:

- at least three out-of-sample folds are available;
- mean out-of-sample rank IC is positive;
- mean top-minus-bottom future return spread is positive.

If no approved model exists, the application refuses ML mode and retains the interpretable linear baseline.

## Google Cloud Run

See [DEPLOY.md](DEPLOY.md) and [deploy/gcp/README.md](deploy/gcp/README.md). The repository includes a PowerShell deployment script that builds one image and deploys it as both a Cloud Run service and a Cloud Run Job.

## Project structure

```text
core/          factor processing and linear scoring
data/          Tushare and other provider adapters
research/      snapshot storage, collection, training, inference, real backtest
backtest/      deterministic synthetic pipeline checks
ml/            advanced optional research components
tests/         regression and research-pipeline tests
deploy/gcp/    Cloud Run deployment assets
docs/          project reports
streamlit_app.py
```

## Research boundaries

- Tushare `bak_basic` is a practical historical source, but it is not a substitute for a licensed, filing-release-aware point-in-time fundamentals database.
- The stored-snapshot backtest starts when the collector is first run; retroactively requested provider fields may contain revision or survivorship risk.
- Analyst expectation factors remain unavailable without a licensed source.
- The system produces research rankings, not orders. Portfolio construction, risk limits, liquidity, compliance, and execution remain outside the scope.

## Documentation

- [Complete v3.4 project report](docs/Mini-GRP-v34-Complete-Project-Report.md)
- [Deployment guide](DEPLOY.md)

## Disclaimer

For research, education, and interview demonstration only. Historical analysis does not represent live investment performance.
