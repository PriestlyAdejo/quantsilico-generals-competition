# Direct WSL job launcher with owned-process recording (avoids nested -Command parsing bugs).
param(
  [Parameter(Mandatory = $true)][string]$JobId,
  [Parameter(Mandatory = $true)][string]$Stage,
  [Parameter(Mandatory = $true)][string]$ScriptPath,
  [int]$WallSeconds = 7200,
  [int]$TransitionLimit = 0,
  [string]$Distro = "Ubuntu"
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
$LogDir = Join-Path $Root "experiments\logs\owned_jobs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Stdout = Join-Path $LogDir "${JobId}_${Stamp}.out.log"
$Stderr = Join-Path $LogDir "${JobId}_${Stamp}.err.log"
$Heartbeat = Join-Path $LogDir "${JobId}_${Stamp}.heartbeat"
$Manifest = Join-Path $Root "experiments\manifests\competition_native_jax_owned_processes.json"
$start = (Get-Date).ToUniversalTime().ToString("o")

$full = "wsl.exe -d $Distro -- bash $ScriptPath"
$p = Start-Process -FilePath "wsl.exe" -ArgumentList @("-d", $Distro, "--", "bash", $ScriptPath) -WorkingDirectory $Root -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru -WindowStyle Hidden

$job = @{
  job_id = $JobId
  stage = $Stage
  windows_pid = $p.Id
  wsl_linux_pid = $null
  full_command = $full
  start_time_utc = $start
  wall_time_limit_s = $WallSeconds
  transition_limit = $TransitionLimit
  stdout_path = ($Stdout -replace "\\", "/")
  stderr_path = ($Stderr -replace "\\", "/")
  heartbeat_path = ($Heartbeat -replace "\\", "/")
  last_checkpoint = $null
  resume_command = "powershell -ExecutionPolicy Bypass -File scripts/windows/start_owned_wsl_script.ps1 -JobId $JobId -Stage $Stage -ScriptPath $ScriptPath -WallSeconds $WallSeconds"
  termination_command = "Stop-Process -Id $($p.Id) -Force"
  final_exit_code = $null
  final_reason = "RUNNING"
  status = "RUNNING"
}

$jobs = @($job)
if (Test-Path $Manifest) {
  try {
    $prev = Get-Content $Manifest -Raw | ConvertFrom-Json
    $kept = @($prev.jobs | Where-Object { $_.status -ne "RUNNING" -or $_.job_id -ne $JobId })
    $jobs = @($kept) + @($job)
  } catch {}
}
@{
  schema_version = 1
  kind = "COMPETITION_NATIVE_JAX_OWNED_PROCESSES"
  updated_at = $start
  jobs = $jobs
} | ConvertTo-Json -Depth 8 | Set-Content -Path $Manifest -Encoding UTF8

Write-Output "STARTED job=$JobId windows_pid=$($p.Id)"
$deadline = (Get-Date).AddSeconds($WallSeconds)
while (-not $p.HasExited -and (Get-Date) -lt $deadline) {
  (Get-Date).ToUniversalTime().ToString("o") | Set-Content -Path $Heartbeat -Encoding UTF8
  Start-Sleep -Seconds 15
}
if (-not $p.HasExited) {
  Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
  $reason = "WALL_TIME_EXCEEDED"
  $code = -1
} else {
  $reason = "EXITED"
  $code = $p.ExitCode
}

$doc2 = Get-Content $Manifest -Raw | ConvertFrom-Json
foreach ($j in $doc2.jobs) {
  if ($j.job_id -eq $JobId -and $j.status -eq "RUNNING") {
    $j.final_exit_code = $code
    $j.final_reason = $reason
    $j.status = if ($code -eq 0) { "COMPLETED" } else { "FAILED" }
  }
}
$doc2.updated_at = (Get-Date).ToUniversalTime().ToString("o")
$doc2 | ConvertTo-Json -Depth 8 | Set-Content -Path $Manifest -Encoding UTF8
Write-Output "FINISHED job=$JobId exit=$code reason=$reason"
exit $(if ($null -eq $code) { 1 } else { $code })
