# Mini-GRP v3.4 Deployment Guide

## Recommended production shape

Use three managed Google Cloud components:

1. **Cloud Run service**: runs Streamlit and approved-model inference.
2. **Cloud Run Job**: incrementally collects snapshots and retrains candidates.
3. **Cloud SQL for PostgreSQL**: stores snapshots and serialized model artifacts.

Store `TUSHARE_TOKEN` and `DATABASE_URL` in Secret Manager. The proxy endpoint is non-secret configuration.

## Local verification

```powershell
pip install -r requirements-dev.txt
python -m compileall -q .
pytest
streamlit run streamlit_app.py
```

Docker Desktop must be running before container checks:

```powershell
docker build -t mini-grp-v34 .
docker run --rm -p 8080:8080 -e PORT=8080 mini-grp-v34
```

Open `http://localhost:8080` and verify all five tabs.

## Google Cloud prerequisites

- A Google Cloud project with billing enabled.
- Google Cloud CLI authenticated with `gcloud auth login`.
- Cloud SQL for PostgreSQL database and user.
- Secret Manager secrets named `mini-grp-tushare-token` and `mini-grp-database-url`.

Use this Cloud SQL URL form; URL-encode special characters in the password:

```text
postgresql+psycopg://mini_grp:PASSWORD@/mini_grp?host=/cloudsql/PROJECT:REGION:INSTANCE
```

## Deploy

From the repository root:

```powershell
.\deploy\gcp\deploy.ps1 -ProjectId YOUR_PROJECT_ID -Region asia-east1
```

The script enables required APIs, creates an Artifact Registry repository when missing, builds the image, deploys the web service, and deploys the research job.

Run the initial 60-month pipeline:

```powershell
gcloud run jobs execute mini-grp-research --region asia-east1 --wait
```

The script also creates or updates a dedicated runtime service account and a monthly Cloud Scheduler trigger. The default schedule is `0 20 1 * *` in timezone `Asia/Shanghai`.

## Current live deployment

As of 2026-06-13, the reference deployment is running in project `project-00980766-847f-47d3-b03`, region `asia-east1`:

| Resource | Current value |
|---|---|
| Web service | [mini-grp-web](https://mini-grp-web-l4pzrl64jq-de.a.run.app) |
| Research job | `mini-grp-research` |
| Cloud SQL | PostgreSQL 16, instance `mini-grp-postgres`, database `mini_grp` |
| Runtime identity | `mini-grp-runtime@project-00980766-847f-47d3-b03.iam.gserviceaccount.com` |
| Scheduler | `mini-grp-monthly`, 20:00 Asia/Shanghai on the first day of each month |
| Stored research data | 60 snapshot dates, 90,000 rows, 58 labelled periods |
| Model state | one Gradient Boosting candidate, available in Experimental ML mode; no approved model |
| Factor audit | 15 features, 58 labelled periods, 86,901 observations; no factor passed the strict stability gate |

The web health endpoint and all five Streamlit tabs have been verified. The stored-snapshot backtest and single-factor audit also run successfully from the deployed web service.

## Runtime configuration

| Name | Location | Purpose |
|---|---|---|
| `TUSHARE_TOKEN` | Secret Manager | provider credential |
| `DATABASE_URL` | Secret Manager | Cloud SQL connection string |
| `TUSHARE_API_URL` | environment variable | compatible proxy endpoint |
| `PORT` | injected by Cloud Run | Streamlit listening port |

## Operational checks

- Cloud Run health endpoint returns success.
- `Model registry` shows snapshot counts and an approved model or a clear baseline-only state.
- `Factor research` rebuilds the labelled panel and reports IC/ICIR, quintile spread, turnover, correlation, and decay without exposing secrets.
- Research mode fails visibly when the provider is unavailable.
- The scheduled job exits with code zero and refreshes only missing/recent dates.
- Cloud SQL backups and point-in-time recovery are enabled.
- Cloud SQL deletion protection is enabled.
- Secret values never appear in logs, source files, or screenshots.

## Capacity notes

The web service is configured for 2 vCPU and 2 GiB RAM. The research job uses 2 vCPU, 4 GiB RAM, and a one-hour task timeout. Increase the job timeout or reduce `--max-stocks` if provider latency makes a 60-month initial load exceed the limit.

## Troubleshooting

**Docker client works but engine does not**

Ensure WSL 2 and Virtual Machine Platform are enabled, restart Windows, then launch Docker Desktop.

**No approved model**

Approval is intentionally withheld when rank IC or top-bottom spread is non-positive. The current 58-period training set still has a negative candidate spread, so the linear model remains the production default. The candidate can be run explicitly through `Experimental ML candidate` for research comparison, with its failed validation metrics displayed on screen.

**Cloud Run cannot connect to PostgreSQL**

Confirm the service and job both have the Cloud SQL instance attached and the runtime service account has `Cloud SQL Client` permission.

**Tushare proxy errors**

Use the proxy token together with its supplied `TUSHARE_API_URL`. Sending a proxy credential to the official endpoint can fail authentication.
