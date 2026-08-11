$ErrorActionPreference = "Continue"
$Root = "C:\Users\pries\Documents\Projects\quantsilico-generals-competition"
Set-Location $Root
$LogDir = Join-Path $Root "experiments\logs\owned_jobs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Stdout = Join-Path $LogDir "v42_pipeline_$Stamp.out.log"
$Stderr = Join-Path $LogDir "v42_pipeline_$Stamp.err.log"
$Heartbeat = Join-Path $LogDir "v42_pipeline_$Stamp.heartbeat"
$ScriptPath = "/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition/scripts/wsl/_run_v4_2_pipeline.sh"
$p = Start-Process -FilePath "wsl.exe" -ArgumentList @("-d", "Ubuntu", "--", "bash", $ScriptPath) -WorkingDirectory $Root -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru -WindowStyle Hidden
$start = (Get-Date).ToUniversalTime().ToString("o")
$job = @{
  job_id = "v42_pipeline"
  stage = "v42_full"
  windows_pid = $p.Id
  wsl_linux_pid = $null
  full_command = "wsl.exe -d Ubuntu -- bash $ScriptPath"
  start_time_utc = $start
  wall_time_limit_s = 28800
  transition_limit = 0
  stdout_path = ($Stdout -replace "\\", "/")
  stderr_path = ($Stderr -replace "\\", "/")
  heartbeat_path = ($Heartbeat -replace "\\", "/")
  last_checkpoint = $null
  resume_command = "powershell -ExecutionPolicy Bypass -File scripts/windows/start_v42_pipeline.ps1"
  termination_command = "Stop-Process -Id $($p.Id) -Force"
  final_exit_code = $null
  final_reason = "RUNNING"
  status = "RUNNING"
}
@{
  schema_version = 1
  kind = "COMPETITION_NATIVE_JAX_OWNED_PROCESSES"
  updated_at = $start
  jobs = @($job)
} | ConvertTo-Json -Depth 8 | Set-Content -Path (Join-Path $Root "experiments\manifests\competition_native_jax_owned_processes.json") -Encoding UTF8
Write-Output "STARTED job=v42_pipeline windows_pid=$($p.Id)"
Write-Output "stdout=$Stdout"
# Heartbeat loop (supervisor)
$deadline = (Get-Date).AddSeconds(28800)
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
$doc2 = Get-Content (Join-Path $Root "experiments\manifests\competition_native_jax_owned_processes.json") -Raw | ConvertFrom-Json
foreach ($j in $doc2.jobs) {
  if ($j.job_id -eq "v42_pipeline" -and $j.status -eq "RUNNING") {
    $j.final_exit_code = $code
    $j.final_reason = $reason
    $j.status = if ($code -eq 0) { "COMPLETED" } else { "FAILED" }
  }
}
$doc2.updated_at = (Get-Date).ToUniversalTime().ToString("o")
$doc2 | ConvertTo-Json -Depth 8 | Set-Content -Path (Join-Path $Root "experiments\manifests\competition_native_jax_owned_processes.json") -Encoding UTF8
Write-Output "FINISHED job=v42_pipeline exit=$code reason=$reason"
