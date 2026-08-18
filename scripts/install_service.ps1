#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Install StockTradingBot as a Windows scheduled task that runs `python main.py live`.

.DESCRIPTION
    Creates (or replaces) a daily-reboot-resilient Windows Scheduled Task named
    'StockTradingBot-Live' that launches the bot's live engine under the
    currently signed-in user account, with the working directory set to the
    repo root so relative paths (logs/, data/, .env) resolve correctly.

    The task:
      * Runs AT STARTUP so the bot auto-starts when the workstation reboots.
      * Restarts on failure with a 60-second delay (max 3 attempts).
      * Runs only when the network is available (best-effort).
      * Executes: python main.py live
      * Logs stdout/stderr to logs/service_stdout.log and logs/service_stderr.log.

.PARAMETER RepoRoot
    Absolute path to the StockTradingBot checkout. Defaults to the directory
    two levels above this script.

.PARAMETER PythonExe
    Absolute path to the Python interpreter. Defaults to `python` on PATH.

.PARAMETER TaskName
    Scheduled task name. Default: 'StockTradingBot-Live'.

.PARAMETER StartAt
    Optional local time to run once per day, e.g. '09:30'. When supplied, the
    task also fires on a daily trigger at that time in addition to startup.

.EXAMPLE
    PS> .\scripts\install_service.ps1
    PS> .\scripts\install_service.ps1 -StartAt '09:30'

.NOTES
    Re-running this script is idempotent: existing tasks with the same name are
    deleted first. Run .\scripts\uninstall_service.ps1 to remove.
#>
[CmdletBinding()]
param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$PythonExe = 'python',
    [string]$TaskName = 'StockTradingBot-Live',
    [string]$StartAt = ''
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $RepoRoot)) {
    throw "RepoRoot not found: $RepoRoot"
}

# Ensure logs/ exists for the scheduled task's stdout/stderr sinks.
$logsDir = Join-Path $RepoRoot 'logs'
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
}

$stdoutLog = Join-Path $logsDir 'service_stdout.log'
$stderrLog = Join-Path $logsDir 'service_stderr.log'

# Action XML: python main.py live with stdout/stderr redirected.
$actionArgs = @(
    '-NoProfile', '-NonInteractive', '-Command',
    "Set-Location -LiteralPath '$RepoRoot'; & '$PythonExe' main.py live 2>> '$stderrLog' >> '$stdoutLog'"
)

$triggersXml = @(
    '<BootTrigger><Enabled>true</Enabled></BootTrigger>'
)
if ($StartAt) {
    $triggersXml += "<CalendarTrigger><Repetition><Interval>PT1H</Interval><StopAtDurationEnd>true</StopAtDurationEnd></Repetition><StartBoundary>2025-01-01T$StartAt</StartBoundary><Enabled>true</Enabled><ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay></CalendarTrigger>"
}

$triggersBlock = ($triggersXml -join '')

$principalId = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value

$taskXml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>StockTradingBot live engine — runs `python main.py live` at startup (and optionally daily at the configured time).</Description>
    <Author>StockTradingBot</Author>
  </RegistrationInfo>
  <Triggers>
    $triggersBlock
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>$principalId</UserId>
      <LogonType>Interactive</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>5</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>powershell.exe</Command>
      <Arguments>$($actionArgs -join ' ')</Arguments>
      <WorkingDirectory>$RepoRoot</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"@

# Replace existing task if present (idempotent install).
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing scheduled task '$TaskName'..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false | Out-Null
}

Write-Host "Registering scheduled task '$TaskName' (repo: $RepoRoot)..." -ForegroundColor Cyan
Register-ScheduledTask -TaskName $TaskName -Xml $taskXml | Out-Null

# Restart-on-failure settings (max 3 attempts, 60s apart).
$task = Get-ScheduledTask -TaskName $TaskName
$task.Settings.RestartCount = 3
$task.Settings.RestartInterval = 'PT1M'
$task | Set-ScheduledTask | Out-Null

Write-Host ""
Write-Host "[OK] Scheduled task '$TaskName' installed." -ForegroundColor Green
Write-Host "    Trigger : AT STARTUP" -ForegroundColor Gray
if ($StartAt) {
    Write-Host "             + daily at $StartAt" -ForegroundColor Gray
}
Write-Host "    Action  : $PythonExe main.py live" -ForegroundColor Gray
Write-Host "    Workdir : $RepoRoot" -ForegroundColor Gray
Write-Host "    Stdout  : $stdoutLog" -ForegroundColor Gray
Write-Host "    Stderr  : $stderrLog" -ForegroundColor Gray
Write-Host ""
Write-Host "Useful commands:" -ForegroundColor DarkGray
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor DarkGray
Write-Host "  Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo" -ForegroundColor DarkGray
Write-Host "  Remove-ScheduledTask -TaskName '$TaskName'  # use uninstall_service.ps1 instead" -ForegroundColor DarkGray