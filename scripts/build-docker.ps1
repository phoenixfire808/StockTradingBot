# ─────────────────────────────────────────────────────────────────────────────
# Build the StockTradingBot Docker image (PowerShell variant for Windows).
#
# Usage:
#     .\scripts\build-docker.ps1                  # build stocktradingbot:latest
#     .\scripts\build-docker.ps1 -Tag v1.2.3      # build stocktradingbot:v1.2.3
#     .\scripts\build-docker.ps1 -NoCache         # force rebuild
# ─────────────────────────────────────────────────────────────────────────────
[CmdletBinding()]
param(
    [string]$ImageName = 'stocktradingbot',
    [string]$Tag = 'latest',
    [switch]$NoCache
)

$ErrorActionPreference = 'Stop'

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "docker CLI not found on PATH. Install Docker Desktop first."
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = (Resolve-Path (Join-Path $ScriptDir '..')).Path

$cacheFlag = ''
if ($NoCache) { $cacheFlag = '--no-cache' }

Write-Host "Building ${ImageName}:${Tag} from ${RepoRoot} ..." -ForegroundColor Cyan
docker build @cacheFlag -t "${ImageName}:${Tag}" -f (Join-Path $RepoRoot 'Dockerfile') $RepoRoot

if ($LASTEXITCODE -ne 0) {
    throw "docker build failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "[OK] Image built: ${ImageName}:${Tag}" -ForegroundColor Green
docker images "${ImageName}:${Tag}" --format "table {{.Repository}}`t{{.Tag}}`t{{.Size}}`t{{.CreatedSince}}"