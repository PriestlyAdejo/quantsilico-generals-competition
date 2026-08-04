#Requires -Version 5.1
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location $Root
$py = Join-Path $Root '.venv-training\Scripts\python.exe'
if (-not (Test-Path $py)) {
  throw 'missing .venv-training — dashboard runs from the training environment'
}
$frontend = Join-Path $Root 'dashboard\frontend'
if (-not (Test-Path (Join-Path $frontend 'node_modules'))) {
  throw 'missing dashboard/frontend/node_modules — run pnpm install there first'
}
$dist = Join-Path $frontend 'dist'
if (-not (Test-Path $dist)) {
  Write-Host 'Building frontend…'
  Push-Location $frontend
  pnpm run build
  Pop-Location
}
$env:PYTHONPATH = Join-Path $Root 'src'
$varDir = Join-Path $Root 'var\dashboard'
New-Item -ItemType Directory -Force -Path $varDir | Out-Null
$pidFile = Join-Path $varDir 'backend.pid'
if (Test-Path $pidFile) {
  $existing = [int](Get-Content $pidFile | Select-Object -First 1)
  try {
    Get-Process -Id $existing -ErrorAction Stop | Out-Null
    try {
      $h = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/health' -TimeoutSec 2
      if ($h.status -eq 'ok' -and $h.bind -eq '127.0.0.1') {
        Write-Host "Dashboard already running (PID $existing). Open http://127.0.0.1:8765"
        exit 0
      }
    } catch {}
  } catch {
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
  }
}
Write-Host 'Starting dashboard on http://127.0.0.1:8765 using .venv-training'
$proc = Start-Process -FilePath $py -ArgumentList @(
  '-m', 'uvicorn', 'dashboard.backend.app.main:app',
  '--host', '127.0.0.1', '--port', '8765', '--app-dir', $Root
) -PassThru -WindowStyle Hidden
$proc.Id | Set-Content $pidFile
Write-Host "PID $($proc.Id) written to var/dashboard/backend.pid"
Write-Host 'Open http://127.0.0.1:8765'
