# ─────────────────────────────────────────────────────────────────────────────
# Run the StockTradingBot Docker container (PowerShell variant for Windows).
#
# Modes (default: dashboard):
#     .\scripts\run-docker.ps1                     # start dashboard on :8501
#     .\scripts\run-docker.ps1 -Mode dry-run       # one-shot dry-run engine
#     .\scripts\run-docker.ps1 -Mode live          # live engine (interactive)
#     .\scripts\run-docker.ps1 -Mode backtest      # run a backtest
#     .\scripts\run-docker.ps1 -Mode bash          # drop into a shell
# ─────────────────────────────────────────────────────────────────────────────
[CmdletBinding()]
param(
    [ValidateSet('ui', 'dashboard', 'dry-run', 'live', 'backtest', 'bash', 'shell')]
    [string]$Mode = 'ui',
    [string]$ImageName = 'stocktradingbot',
    [string]$Tag = 'latest'
)

$ErrorActionPreference = 'Stop'

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "docker CLI not found on PATH. Install Docker Desktop first."
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = (Resolve-Path (Join-Path $ScriptDir '..')).Path

foreach ($d in @('logs', 'data', 'reports')) {
    if (-not (Test-Path (Join-Path $RepoRoot $d))) {
        New-Item -ItemType Directory -Path (Join-Path $RepoRoot $d) -Force | Out-Null
    }
}

$envFile = Join-Path $RepoRoot '.env'
$envArgs = @()
if (Test-Path $envFile) { $envArgs += @('--env-file', $envFile) }

$mounts = @(
    '-v', "${RepoRoot}\logs:/app/logs",
    '-v', "${RepoRoot}\data:/app/data",
    '-v', "${RepoRoot}\reports:/app/reports"
)

switch ($Mode) {
    { $_ -in 'ui', 'dashboard' } {
        Write-Host "Starting Streamlit dashboard on http://localhost:8501 ..." -ForegroundColor Cyan
        docker run --rm -p 8501:8501 @mounts @envArgs "${ImageName}:${Tag}" `
            python main.py ui --port 8501
    }
    'dry-run' {
        Write-Host "Starting dry-run engine (Ctrl-C to stop) ..." -ForegroundColor Cyan
        docker run --rm -p 8501:8501 -it @mounts @envArgs "${ImageName}:${Tag}" `
            python main.py dry-run
    }
    'live' {
        Write-Host "Starting LIVE engine (interactive) ..." -ForegroundColor Cyan
        docker run --rm -p 8501:8501 -it @mounts @envArgs "${ImageName}:${Tag}" `
            python main.py live
    }
    'backtest' {
        Write-Host "Running backtest ..." -ForegroundColor Cyan
        docker run --rm -p 8501:8501 @mounts @envArgs "${ImageName}:${Tag}" `
            python main.py backtest
    }
    { $_ -in 'bash', 'shell' } {
        Write-Host "Opening shell in ${ImageName}:${Tag} ..." -ForegroundColor Cyan
        docker run --rm -p 8501:8501 -it --entrypoint bash @mounts @envArgs "${ImageName}:${Tag}"
    }
}