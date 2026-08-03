#Requires -Version 5.1
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$pidFile = Join-Path $Root 'var\dashboard\backend.pid'
if (-not (Test-Path $pidFile)) {
  Write-Host 'No dashboard PID file found.'
  exit 0
}
$procId = [int](Get-Content $pidFile | Select-Object -First 1)
try {
  Stop-Process -Id $procId -Force -ErrorAction Stop
  Write-Host "Stopped dashboard PID $procId"
} catch {
  Write-Host "Process $procId not running"
}
Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
