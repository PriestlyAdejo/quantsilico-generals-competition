#!/usr/bin/env bash
# Bootstrap CUDA JAX venv for QuantSilico inside WSL2 Ubuntu.
# PYTHON_COMPATIBILITY_RULE: do not use unsupported system Python; never modify system Python.
# Prefer user-space interpreters (uv) so passwordless sudo is not required.
set -euo pipefail
REPO_ROOT="${1:-}"
if [[ -z "${REPO_ROOT}" ]]; then
  echo "usage: bootstrap_quantsilico_jax_gpu.sh <repo_root_wsl_path>"
  exit 1
fi
LOG_DIR="${REPO_ROOT}/experiments/logs/wsl_jax_bootstrap"
mkdir -p "${LOG_DIR}"
{
  echo "distro=$(. /etc/os-release && echo $PRETTY_NAME)"
  uname -a
  python3 --version || true
  nvidia-smi || true
} | tee "${LOG_DIR}/linux_precheck.txt"

VENV="${HOME}/.venvs/quantsilico-jax-gpu"
export PATH="${HOME}/.local/bin:${PATH}"

need_user_python() {
  local major minor
  major="$(python3 -c 'import sys; print(sys.version_info[0])' 2>/dev/null || echo 0)"
  minor="$(python3 -c 'import sys; print(sys.version_info[1])' 2>/dev/null || echo 0)"
  # Official CUDA JAX wheels currently cover ~3.10–3.13; 3.14 is unsupported.
  if [[ "${major}" -eq 3 && "${minor}" -ge 10 && "${minor}" -le 13 ]]; then
    return 1
  fi
  return 0
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi
  echo "Installing uv into \$HOME (no sudo)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
}

PY_BIN=""
if command -v python3.12 >/dev/null 2>&1; then
  PY_BIN="python3.12"
elif command -v python3.11 >/dev/null 2>&1; then
  PY_BIN="python3.11"
elif command -v python3.13 >/dev/null 2>&1; then
  PY_BIN="python3.13"
elif ! need_user_python; then
  PY_BIN="python3"
else
  ensure_uv
  echo "Installing CPython 3.12 via uv (system python unsupported for JAX CUDA wheels)"
  uv python install 3.12
  PY_BIN="$(uv python find 3.12)"
fi

echo "Using interpreter ${PY_BIN} ($("${PY_BIN}" --version 2>&1))"
echo "${PY_BIN}" > "${LOG_DIR}/selected_python.txt"
"${PY_BIN}" --version | tee "${LOG_DIR}/selected_python_version.txt"

rm -rf "${VENV}"
if command -v uv >/dev/null 2>&1; then
  uv venv "${VENV}" --python "${PY_BIN}" --seed
else
  "${PY_BIN}" -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
if ! python -m pip --version >/dev/null 2>&1; then
  python -m ensurepip --upgrade || uv pip install --python "${VENV}/bin/python" pip setuptools wheel
fi
python -m pip install --upgrade pip setuptools wheel

CUDA_EXTRA="cuda12"
if ! python -m pip install --upgrade "jax[${CUDA_EXTRA}]"; then
  CUDA_EXTRA="cuda13"
  python -m pip install --upgrade "jax[${CUDA_EXTRA}]"
fi
python -m pip install --upgrade optax
cd "${REPO_ROOT}"
python -m pip install -e ".[dev]" || python -m pip install -e .

python - <<PY
import json, platform, subprocess
from pathlib import Path
import jax
try:
    import jaxlib
    jaxlib_v = jaxlib.__version__
except Exception as e:
    jaxlib_v = f"error:{e}"
try:
    import optax
    optax_v = optax.__version__
except Exception as e:
    optax_v = f"error:{e}"
info = {
  "python": platform.python_version(),
  "python_executable": __import__("sys").executable,
  "jax": jax.__version__,
  "jaxlib": jaxlib_v,
  "optax": optax_v,
  "cuda_wheel_extra": "${CUDA_EXTRA}",
  "backend": jax.default_backend(),
  "devices": [str(d) for d in jax.devices()],
}
freeze = subprocess.check_output(["pip", "freeze"], text=True)
Path("experiments/manifests").mkdir(parents=True, exist_ok=True)
Path("experiments/manifests/competition_native_jax_wsl_environment.json").write_text(
  json.dumps({"schema_version": 1, "kind": "WSL_JAX_ENV", "jax": info, "pip_freeze": freeze.splitlines()}, indent=2) + "\n",
  encoding="utf-8",
)
print(json.dumps(info, indent=2))
PY
