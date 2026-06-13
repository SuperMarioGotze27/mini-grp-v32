param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [string]$Region = "asia-east1",
    [string]$Repository = "mini-grp",
    [string]$ServiceName = "mini-grp-web",
    [string]$JobName = "mini-grp-research",
    [string]$CloudSqlInstance = "mini-grp-postgres",
    [string]$RuntimeServiceAccount = "mini-grp-runtime",
    [string]$SchedulerName = "mini-grp-monthly",
    [string]$Schedule = "0 20 1 * *",
    [string]$TimeZone = "Asia/Shanghai",
    [string]$ImageTag = "latest"
)

$ErrorActionPreference = "Stop"
$GcloudCommand = Get-Command gcloud.cmd -ErrorAction SilentlyContinue
if (-not $GcloudCommand) {
    $GcloudCommand = Get-Command gcloud -ErrorAction Stop
}
$Gcloud = $GcloudCommand.Source
$Image = "$Region-docker.pkg.dev/$ProjectId/$Repository/mini-grp:$ImageTag"
$InstanceConnection = "$ProjectId`:$Region`:$CloudSqlInstance"
$RuntimeServiceAccountEmail = "$RuntimeServiceAccount@$ProjectId.iam.gserviceaccount.com"
$SchedulerUri = "https://run.googleapis.com/v2/projects/$ProjectId/locations/$Region/jobs/$JobName`:run"

& $Gcloud config set project $ProjectId
& $Gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com sqladmin.googleapis.com secretmanager.googleapis.com cloudscheduler.googleapis.com iam.googleapis.com

$runtimeAccount = & $Gcloud iam service-accounts describe $RuntimeServiceAccountEmail --format="value(email)" 2>$null
if (-not $runtimeAccount) {
    & $Gcloud iam service-accounts create $RuntimeServiceAccount --display-name="Mini-GRP runtime"
}

foreach ($role in @("roles/cloudsql.client", "roles/secretmanager.secretAccessor", "roles/logging.logWriter")) {
    & $Gcloud projects add-iam-policy-binding $ProjectId `
        --member="serviceAccount:$RuntimeServiceAccountEmail" `
        --role=$role `
        --quiet | Out-Null
}

foreach ($secret in @("mini-grp-tushare-token", "mini-grp-database-url")) {
    $existingSecret = & $Gcloud secrets describe $secret --format="value(name)" 2>$null
    if (-not $existingSecret) {
        throw "Required Secret Manager secret '$secret' does not exist. Create it before deployment."
    }
}

$existingInstance = & $Gcloud sql instances describe $CloudSqlInstance --format="value(name)" 2>$null
if (-not $existingInstance) {
    throw "Required Cloud SQL instance '$CloudSqlInstance' does not exist. Create it before deployment."
}

$existingRepo = & $Gcloud artifacts repositories describe $Repository --location $Region --format="value(name)" 2>$null
if (-not $existingRepo) {
    & $Gcloud artifacts repositories create $Repository --repository-format=docker --location=$Region
}

$BuildServiceAccount = & $Gcloud builds get-default-service-account --project $ProjectId
foreach ($role in @("roles/cloudbuild.builds.builder", "roles/artifactregistry.writer")) {
    & $Gcloud projects add-iam-policy-binding $ProjectId `
        --member="serviceAccount:$BuildServiceAccount" `
        --role=$role `
        --quiet | Out-Null
}

& $Gcloud builds submit --tag $Image .

& $Gcloud run deploy $ServiceName `
    --image $Image `
    --region $Region `
    --platform managed `
    --allow-unauthenticated `
    --service-account $RuntimeServiceAccountEmail `
    --port 8080 `
    --cpu 2 `
    --memory 2Gi `
    --timeout 300 `
    --min 0 `
    --max 3 `
    --add-cloudsql-instances $InstanceConnection `
    --set-secrets "TUSHARE_TOKEN=mini-grp-tushare-token:latest,DATABASE_URL=mini-grp-database-url:latest" `
    --set-env-vars "TUSHARE_API_URL=https://ts.gyzcloud.top/api" `
    --quiet

& $Gcloud run jobs deploy $JobName `
    --image $Image `
    --region $Region `
    --service-account $RuntimeServiceAccountEmail `
    --command python `
    --args=-m,research.cli,pipeline,--months,60,--max-stocks,1500,--overlay-weight,0.15 `
    --cpu 2 `
    --memory 4Gi `
    --task-timeout 3600s `
    --max-retries 1 `
    --set-cloudsql-instances $InstanceConnection `
    --set-secrets "TUSHARE_TOKEN=mini-grp-tushare-token:latest,DATABASE_URL=mini-grp-database-url:latest" `
    --set-env-vars "TUSHARE_API_URL=https://ts.gyzcloud.top/api" `
    --quiet

& $Gcloud run jobs add-iam-policy-binding $JobName `
    --region $Region `
    --member="serviceAccount:$RuntimeServiceAccountEmail" `
    --role="roles/run.invoker" `
    --quiet | Out-Null

$existingScheduler = & $Gcloud scheduler jobs describe $SchedulerName --location $Region --format="value(name)" 2>$null
$schedulerArguments = @(
    $SchedulerName,
    "--location=$Region",
    "--schedule=$Schedule",
    "--time-zone=$TimeZone",
    "--uri=$SchedulerUri",
    "--http-method=POST",
    "--oauth-service-account-email=$RuntimeServiceAccountEmail",
    "--oauth-token-scope=https://www.googleapis.com/auth/cloud-platform",
    "--quiet"
)
if ($existingScheduler) {
    & $Gcloud scheduler jobs update http @schedulerArguments
} else {
    & $Gcloud scheduler jobs create http @schedulerArguments
}

Write-Host "Cloud Run service, research job, and monthly scheduler deployed."
Write-Host "Web URL: $(& $Gcloud run services describe $ServiceName --region $Region --format='value(status.url)')"
Write-Host "Execute the initial pipeline with:"
Write-Host "gcloud.cmd run jobs execute $JobName --region $Region --wait"
