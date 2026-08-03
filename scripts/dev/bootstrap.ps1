#Requires -Version 5.1
<#
.SYNOPSIS
  Bootstrap the private Generals competition development environment.

.PARAMETER ScaffoldOnly
  Install only the private package and development tools after a recorded
  official-dependency failure. Does not install the official engine.
#>
[CmdletBinding()]
param(
    [switch]$ScaffoldOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Assert-RepositoryRoot {
    $markers = @('pyproject.toml', 'src\generals_bot', 'third_party\generals-bots')
    foreach ($marker in $markers) {
        if (-not (Test-Path -LiteralPath (Join-Path (Get-Location) $marker))) {
            throw "Run bootstrap.ps1 from the repository root. Missing: $marker"
        }
    }
}

function Resolve-Python312 {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "C:\Program Files\Python312\python.exe"
    )
    $found = Get-ChildItem -Path $candidates -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) {
        return $found.FullName
    }

    try {
        $fromLauncher = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $fromLauncher) {
            return $fromLauncher.Trim()
        }
    } catch {
        # fall through
    }

    throw "Python 3.12 interpreter not found. Install Python 3.12.10 and re-run."
}

function Assert-ExactPythonVersion {
    param([string]$PythonExe)
    $version = & $PythonExe --version 2>&1
    Write-Host "Interpreter: $PythonExe"
    Write-Host "Reported:    $version"
    if ("$version" -ne 'Python 3.12.10') {
        throw "Expected Python 3.12.10, got: $version"
    }
}

Assert-RepositoryRoot

$python = Resolve-Python312
Assert-ExactPythonVersion -PythonExe $python

$venvPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating .venv with $python"
    & $python -m venv .venv
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Failed to create .venv"
}

Write-Host "Using venv: $venvPython"
& $venvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

if ($ScaffoldOnly) {
    Write-Host "=== ScaffoldOnly mode: skipping official competition requirements and engine ==="
    Write-Host "environment_parity: false"
    Write-Host "status: bootstrap-only"
    & $venvPython -m pip install -e ".[dev]"
    if ($LASTEXITCODE -ne 0) { throw "scaffold-only editable install failed" }
    & $venvPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw "pip check failed in scaffold-only mode" }
} else {
    $req = 'third_party\generals-bots\competition\requirements.txt'
    if (-not (Test-Path -LiteralPath $req)) {
        throw "Official requirements missing: $req"
    }

    Write-Host "Installing official competition requirements (authoritative)..."
    & $venvPython -m pip install -r $req
    if ($LASTEXITCODE -ne 0) {
        throw @"
Official competition dependency installation failed.
Record this failure, then re-run with -ScaffoldOnly if you only need
repository scaffolding / lint / package tests.
Do not run the official matchup until dependencies install successfully.
"@
    }

    Write-Host "Installing official engine editable with --no-deps..."
    & $venvPython -m pip install --no-deps -e third_party\generals-bots
    if ($LASTEXITCODE -ne 0) { throw "official engine editable install failed" }

    # Competition requirements are authoritative and omit GUI/live-client deps.
    # Try to install them separately so engine metadata satisfies pip check
    # without re-resolving the competition lock. Failure here is recorded, not fatal
    # for competition-path work.
    Write-Host "Installing engine non-competition extras (pygame, python-socketio)..."
    & $venvPython -m pip install --upgrade-strategy only-if-needed 'pygame>=2.6.0' 'python-socketio[client]>=5.11.4'
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Engine extras (pygame/socketio) failed to install. Competition lock remains authoritative."
    }

    Write-Host "Installing private package with development extras..."
    & $venvPython -m pip install -e ".[dev]"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Editable [dev] install failed; retrying packages individually then --no-build-isolation..."
        $devPkgs = @(
            'pytest>=8.0',
            'pytest-cov>=5.0',
            'hypothesis>=6.100',
            'ruff>=0.6',
            'mypy>=1.11',
            'psutil>=6.0',
            'PyYAML>=6.0'
        )
        foreach ($pkg in $devPkgs) {
            $ok = $false
            for ($i = 1; $i -le 5; $i++) {
                & $venvPython -m pip install $pkg
                if ($LASTEXITCODE -eq 0) { $ok = $true; break }
                Start-Sleep -Seconds 2
            }
            if (-not $ok) { throw "failed to install $pkg after retries" }
        }
        & $venvPython -m pip install --no-build-isolation -e .
        if ($LASTEXITCODE -ne 0) { throw "private package editable install failed" }
    }

    & $venvPython -m pip check
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "pip check reported issues (often missing pygame/socketio when extras install failed)."
    }
}

Write-Host ""
Write-Host "=== Environment versions ==="
& $venvPython --version
& $venvPython -c "import sys; print('executable:', sys.executable)"
& $venvPython -m pip --version
try { & $venvPython -c "import numpy; print('numpy', numpy.__version__)" } catch { Write-Host "numpy: not installed" }
try { & $venvPython -c "import jax; print('jax', jax.__version__)" } catch { Write-Host "jax: not installed" }
try { & $venvPython -c "import generals_bot; print('generals_bot', generals_bot.__version__)" } catch { Write-Host "generals_bot: not installed" }
if (-not $ScaffoldOnly) {
    try { & $venvPython -c "import generals; print('official generals import: ok')" } catch { Write-Host "official generals: not importable" }
}

Write-Host ""
Write-Host "Bootstrap completed."
if ($ScaffoldOnly) {
    Write-Host "Mode: ScaffoldOnly (environment_parity=false)"
} else {
    Write-Host "Mode: full (official requirements + engine --no-deps + private [dev])"
}
