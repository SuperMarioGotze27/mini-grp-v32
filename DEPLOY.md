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

Open `http://localhost:8080` and verify all four tabs.

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

Then add a monthly scheduler trigger from the Cloud Run Job **Triggers** tab. A suitable schedule is `0 20 1 * *` in timezone `Asia/Shanghai`.

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
- Research mode fails visibly when the provider is unavailable.
- The scheduled job exits with code zero and refreshes only missing/recent dates.
- Cloud SQL backups and point-in-time recovery are enabled.
- Secret values never appear in logs, source files, or screenshots.

## Capacity notes

The web service is configured for 2 vCPU and 2 GiB RAM. The research job uses 2 vCPU, 4 GiB RAM, and a one-hour task timeout. Increase the job timeout or reduce `--max-stocks` if provider latency makes a 60-month initial load exceed the limit.

## Troubleshooting

**Docker client works but engine does not**

Ensure WSL 2 and Virtual Machine Platform are enabled, restart Windows, then launch Docker Desktop.

**No approved model**

Collect at least 11 labelled month-end snapshots. Approval is intentionally withheld when rank IC or top-bottom spread is non-positive.

**Cloud Run cannot connect to PostgreSQL**

Confirm the service and job both have the Cloud SQL instance attached and the runtime service account has `Cloud SQL Client` permission.

**Tushare proxy errors**

Use the proxy token together with its supplied `TUSHARE_API_URL`. Sending a proxy credential to the official endpoint can fail authentication.
