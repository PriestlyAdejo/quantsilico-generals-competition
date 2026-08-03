#Requires -Version 5.1
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "missing .venv" }
& $py -m pip install fastapi 'uvicorn[standard]' pydantic -q
$env:PYTHONPATH = (Join-Path $Root 'src')
Write-Host "Starting dashboard on http://127.0.0.1:8765"
& $py -m uvicorn dashboard.backend.app.main:app --host 127.0.0.1 --port 8765 --app-dir $Root
