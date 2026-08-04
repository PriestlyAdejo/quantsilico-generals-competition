#Requires -Version 5.1
$ErrorActionPreference = 'Continue'
Set-StrictMode -Version Latest
$Root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$pidFile = Join-Path $Root 'var\dashboard\backend.pid'

Write-Host "QuantSilico dashboard status"
Write-Host "repo: $Root"

$recorded = $null
if (Test-Path $pidFile) {
  $recorded = [int](Get-Content $pidFile | Select-Object -First 1)
  Write-Host "recorded_pid: $recorded"
} else {
  Write-Host "recorded_pid: (none)"
}

$running = $false
if ($recorded) {
  try {
    $proc = Get-Process -Id $recorded -ErrorAction Stop
    $running = $true
    Write-Host "process_running: true ($($proc.ProcessName))"
  } catch {
    Write-Host "process_running: false"
  }
} else {
  Write-Host "process_running: false"
}

$portOwners = @()
try {
  $portOwners = @(
    Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue |
      Select-Object -ExpandProperty OwningProcess -Unique
  )
} catch {}
if ($portOwners.Count -gt 0) {
  Write-Host ("port_8765_bound: true owners=" + ($portOwners -join ','))
  foreach ($owner in $portOwners) {
    if ($recorded -and $owner -eq $recorded) {
      Write-Host "pid_relationship: port_owner_equals_recorded_launcher_or_backend"
    } elseif ($recorded) {
      try {
        $child = Get-CimInstance Win32_Process -Filter "ProcessId=$owner" -ErrorAction SilentlyContinue
        if ($child -and [int]$child.ParentProcessId -eq $recorded) {
          Write-Host "pid_relationship: port_owner=$owner is child of recorded_pid=$recorded (launcher/parent)"
        } else {
          Write-Host "pid_relationship: port_owner=$owner differs from recorded_pid=$recorded (inspect parent/child)"
        }
      } catch {
        Write-Host "pid_relationship: port_owner=$owner recorded_pid=$recorded (relationship unknown)"
      }
    } else {
      Write-Host "pid_relationship: port_owner=$owner (no recorded pid file)"
    }
  }
} else {
  Write-Host "port_8765_bound: false"
}

$healthOk = $false
try {
  $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/health' -TimeoutSec 3
  if ($health.status -eq 'ok' -and $health.bind -eq '127.0.0.1') {
    $healthOk = $true
    Write-Host "health: ok (QuantSilico bind=$($health.bind) schema=$($health.schema_version))"
    if ($health.PSObject.Properties.Name -contains 'branch') {
      Write-Host "served_branch: $($health.branch)"
    }
    if ($health.PSObject.Properties.Name -contains 'commit') {
      Write-Host "served_commit: $($health.commit)"
    }
  } else {
    Write-Host "health: unexpected payload - not assuming QuantSilico dashboard"
  }
} catch {
  Write-Host "health: unreachable (do not assume any service on port 8765 is QuantSilico)"
}

if ($running -and $healthOk) {
  exit 0
} elseif (-not $running -and -not $healthOk) {
  exit 1
} else {
  exit 2
}
