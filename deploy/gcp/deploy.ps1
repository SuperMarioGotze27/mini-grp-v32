param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [string]$Region = "asia-east1",
    [string]$Repository = "mini-grp",
    [string]$ServiceName = "mini-grp-web",
    [string]$JobName = "mini-grp-research",
    [string]$CloudSqlInstance = "mini-grp-postgres",
    [string]$ImageTag = "latest"
)

$ErrorActionPreference = "Stop"
$Image = "$Region-docker.pkg.dev/$ProjectId/$Repository/mini-grp:$ImageTag"
$InstanceConnection = "$ProjectId`:$Region`:$CloudSqlInstance"

gcloud config set project $ProjectId
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com sqladmin.googleapis.com secretmanager.googleapis.com cloudscheduler.googleapis.com

$existingRepo = gcloud artifacts repositories describe $Repository --location $Region --format="value(name)" 2>$null
if (-not $existingRepo) {
    gcloud artifacts repositories create $Repository --repository-format=docker --location=$Region
}

gcloud builds submit --tag $Image .

gcloud run deploy $ServiceName `
    --image $Image `
    --region $Region `
    --platform managed `
    --allow-unauthenticated `
    --port 8080 `
    --cpu 2 `
    --memory 2Gi `
    --timeout 300 `
    --min 0 `
    --max 3 `
    --add-cloudsql-instances $InstanceConnection `
    --set-secrets "TUSHARE_TOKEN=mini-grp-tushare-token:latest,DATABASE_URL=mini-grp-database-url:latest" `
    --set-env-vars "TUSHARE_API_URL=https://ts.gyzcloud.top/api"

gcloud run jobs deploy $JobName `
    --image $Image `
    --region $Region `
    --command python `
    --args=-m,research.cli,pipeline,--months,60,--max-stocks,1500,--overlay-weight,0.15 `
    --cpu 2 `
    --memory 4Gi `
    --task-timeout 3600s `
    --max-retries 1 `
    --add-cloudsql-instances $InstanceConnection `
    --set-secrets "TUSHARE_TOKEN=mini-grp-tushare-token:latest,DATABASE_URL=mini-grp-database-url:latest" `
    --set-env-vars "TUSHARE_API_URL=https://ts.gyzcloud.top/api"

Write-Host "Cloud Run service and research job deployed."
Write-Host "Execute the initial pipeline with:"
Write-Host "gcloud run jobs execute $JobName --region $Region --wait"
