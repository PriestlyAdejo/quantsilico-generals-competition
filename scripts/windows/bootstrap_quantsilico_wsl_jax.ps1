# Bootstrap Ubuntu WSL2 + invoke Linux CUDA JAX setup for QuantSilico.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/windows/bootstrap_quantsilico_wsl_jax.ps1

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
$LogDir = Join-Path $Root "experiments\logs\wsl_jax_bootstrap"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Out = Join-Path $LogDir "bootstrap_$Stamp.out.log"
$Err = Join-Path $LogDir "bootstrap_$Stamp.err.log"

function Write-Log([string]$msg) {
  $line = "$(Get-Date -Format o) $msg"
  Add-Content -Path $Out -Value $line
  Write-Output $line
}

Write-Log "Windows: $([System.Environment]::OSVersion.VersionString)"
Write-Log "PowerShell: $($PSVersionTable.PSVersion)"
nvidia-smi 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "nvidia_smi_$Stamp.txt")
wsl --status 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "wsl_status_$Stamp.txt")
wsl --list --verbose 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "wsl_list_$Stamp.txt")

$distros = (wsl --list --quiet 2>$null)
$hasUbuntu = $false
if ($distros) {
  foreach ($d in $distros) {
    if ($d -match "Ubuntu") { $hasUbuntu = $true; $UbuntuName = $d.Trim(); break }
  }
}

if (-not $hasUbuntu) {
  Write-Log "No Ubuntu distro found. Attempting: wsl --install -d Ubuntu"
  Write-Log "If elevation is required, set AWAITING_OPERATOR_ACTION and run elevated:"
  Write-Log "  wsl --install -d Ubuntu"
  try {
    wsl --update 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "wsl_update_$Stamp.txt")
    $p = Start-Process -FilePath "wsl.exe" -ArgumentList "--install","-d","Ubuntu" -Wait -PassThru -NoNewWindow
    Write-Log "wsl --install exit=$($p.ExitCode)"
    if ($p.ExitCode -ne 0) {
      Write-Log "STATUS=AWAITING_OPERATOR_ACTION"
      Write-Log "CONTINUATION=wsl --install -d Ubuntu"
      exit 2
    }
  } catch {
    Write-Log "STATUS=AWAITING_OPERATOR_ACTION err=$_"
    Write-Log "CONTINUATION=wsl --install -d Ubuntu"
    exit 2
  }
  Write-Log "STATUS=AWAITING_REBOOT_OR_USER_SETUP"
  Write-Log "CONTINUATION=powershell -ExecutionPolicy Bypass -File scripts/windows/bootstrap_quantsilico_wsl_jax.ps1"
  exit 3
}

$UbuntuName = "Ubuntu"
Write-Log "Using distro=$UbuntuName"
# Convert path for WSL
$wslRoot = (wsl.exe -d $UbuntuName -- bash -lc "wslpath -a '$Root'").Trim()
Write-Log "WSL root=$wslRoot"
wsl.exe --distribution $UbuntuName -- bash -lc "bash '$wslRoot/scripts/wsl/bootstrap_quantsilico_jax_gpu.sh' '$wslRoot'" 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "linux_bootstrap_$Stamp.txt")
$verify = wsl.exe --distribution $UbuntuName -- bash -lc "source `$HOME/.venvs/quantsilico-jax-gpu/bin/activate && python '$wslRoot/scripts/wsl/verify_quantsilico_jax_gpu.py' '$wslRoot'"
Write-Log $verify
Write-Log "bootstrap finished"
