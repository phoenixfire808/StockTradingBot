#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Remove the StockTradingBot scheduled task installed by install_service.ps1.

.DESCRIPTION
    Unregisters the Windows Scheduled Task that launches `python main.py live`.
    If the task is currently running, it is stopped first. The script is safe
    to re-run; it exits 0 if the task does not exist.

.PARAMETER TaskName
    Scheduled task name. Default: 'StockTradingBot-Live'.

.EXAMPLE
    PS> .\scripts\uninstall_service.ps1
#>
[CmdletBinding()]
param(
    [string]$TaskName = 'StockTradingBot-Live'
)

$ErrorActionPreference = 'Stop'

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Host "Scheduled task '$TaskName' is not registered. Nothing to do." -ForegroundColor Yellow
    exit 0
}

try {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue | Out-Null
}
catch {
    # Task may have nothing to stop (e.g., never started). Ignore.
}

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false | Out-Null
Write-Host "[OK] Scheduled task '$TaskName' removed." -ForegroundColor Green