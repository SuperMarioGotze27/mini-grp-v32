# Google Cloud Run deployment

This deployment uses one container image in two roles:

- Cloud Run service: Streamlit screening and model monitoring.
- Cloud Run job: incremental Tushare snapshot collection and walk-forward model training.
- Cloud SQL for PostgreSQL: point-in-time snapshots and model artifacts.
- Secret Manager: `TUSHARE_TOKEN` and `DATABASE_URL`.

Before running `deploy.ps1`, create the Cloud SQL instance, database, user, and these secrets:

```powershell
gcloud secrets create mini-grp-tushare-token --replication-policy=automatic
gcloud secrets versions add mini-grp-tushare-token --data-file=-

gcloud secrets create mini-grp-database-url --replication-policy=automatic
gcloud secrets versions add mini-grp-database-url --data-file=-
```

The database URL should use the Cloud SQL Unix socket:

```text
postgresql+psycopg://mini_grp:URL_ENCODED_PASSWORD@/mini_grp?host=/cloudsql/PROJECT:REGION:INSTANCE
```

Deploy from the repository root:

```powershell
.\deploy\gcp\deploy.ps1 -ProjectId YOUR_PROJECT_ID
```

The deployment script creates the runtime service account, deploys both workloads, and creates or updates the monthly scheduler. Run the initial history build once:

```powershell
gcloud run jobs execute mini-grp-research --region asia-east1 --wait
```

The default scheduler is `mini-grp-monthly`, using `0 20 1 * *` with timezone `Asia/Shanghai`. The collector skips completed history and refreshes the latest two month-ends so newly available forward labels are filled in.

Reference deployment verified on 2026-06-13:

- Web: https://mini-grp-web-l4pzrl64jq-de.a.run.app
- Region: `asia-east1`
- Snapshot data: 60 dates and 90,000 rows
- Model registry: one candidate and no approved model
- Scheduler: enabled, next monthly run controlled by Cloud Scheduler
