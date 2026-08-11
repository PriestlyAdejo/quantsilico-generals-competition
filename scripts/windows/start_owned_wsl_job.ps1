# Long-job supervisor for QuantSilico WSL CUDA daytime resume.
# Usage:
#   powershell -File scripts/windows/start_owned_wsl_job.ps1 -JobId r_d_bootstrap -Stage R_D `
#     -Command "bash '/mnt/.../scripts/wsl/bootstrap_quantsilico_jax_gpu.sh' '/mnt/...'" `
#     -WallSeconds 3600
param(
  [Parameter(Mandatory = $true)][string]$JobId,
  [Parameter(Mandatory = $true)][string]$Stage,
  [Parameter(Mandatory = $true)][string]$Command,
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
$bashCmd = "echo `$`$ > /tmp/${JobId}.pid; while true; do date -Is > '$($Heartbeat -replace '\\','/')' 2>/dev/null || date -Is > /tmp/${JobId}.heartbeat; sleep 15; done & HB=`$!; ($Command); EC=`$?; kill `$HB 2>/dev/null; exit `$EC"
# Simpler: run command directly; heartbeat via Windows side
$arg = @("-d", $Distro, "--", "bash", "-lc", $Command)
$p = Start-Process -FilePath "wsl.exe" -ArgumentList $arg -WorkingDirectory $Root -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru -WindowStyle Hidden

$job = [ordered]@{
  job_id = $JobId
  stage = $Stage
  windows_pid = $p.Id
  wsl_linux_pid = $null
  full_command = "wsl.exe -d $Distro -- bash -lc $Command"
  start_time_utc = $start
  wall_time_limit_s = $WallSeconds
  transition_limit = $TransitionLimit
  stdout_path = $Stdout.Replace("\", "/")
  stderr_path = $Stderr.Replace("\", "/")
  heartbeat_path = $Heartbeat.Replace("\", "/")
  last_checkpoint = $null
  resume_command = "powershell -ExecutionPolicy Bypass -File scripts/windows/start_owned_wsl_job.ps1 -JobId $JobId -Stage $Stage -Command `"$Command`" -WallSeconds $WallSeconds"
  termination_command = "Stop-Process -Id $($p.Id) -Force"
  final_exit_code = $null
  final_reason = "RUNNING"
  status = "RUNNING"
}

# Merge into owned processes manifest
$doc = @{ schema_version = 1; kind = "COMPETITION_NATIVE_JAX_OWNED_PROCESSES"; updated_at = $start; jobs = @() }
if (Test-Path $Manifest) {
  try { $doc = Get-Content $Manifest -Raw | ConvertFrom-Json } catch {}
}
$jobs = @()
if ($doc.jobs) { $jobs = @($doc.jobs | Where-Object { $_.job_id -ne $JobId }) }
$jobs += $job
$outDoc = @{
  schema_version = 1
  kind = "COMPETITION_NATIVE_JAX_OWNED_PROCESSES"
  updated_at = $start
  jobs = $jobs
}
($outDoc | ConvertTo-Json -Depth 8) | Set-Content -Path $Manifest -Encoding UTF8

Write-Output "STARTED job=$JobId windows_pid=$($p.Id) stdout=$Stdout"
# Heartbeat loop until exit or wall
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

# Update manifest final fields
$doc2 = Get-Content $Manifest -Raw | ConvertFrom-Json
foreach ($j in $doc2.jobs) {
  if ($j.job_id -eq $JobId) {
    $j.final_exit_code = $code
    $j.final_reason = $reason
    $j.status = if ($code -eq 0) { "COMPLETED" } else { "FAILED" }
  }
}
$doc2.updated_at = (Get-Date).ToUniversalTime().ToString("o")
# Keep only non-RUNNING completed history + clear RUNNING
($doc2 | ConvertTo-Json -Depth 8) | Set-Content -Path $Manifest -Encoding UTF8
Write-Output "FINISHED job=$JobId exit=$code reason=$reason"
exit $code
