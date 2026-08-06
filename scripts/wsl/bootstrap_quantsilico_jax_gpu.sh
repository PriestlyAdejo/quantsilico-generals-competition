#!/usr/bin/env bash
# Bootstrap CUDA JAX venv for QuantSilico inside WSL2 Ubuntu.
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

sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip git build-essential

VENV="${HOME}/.venvs/quantsilico-jax-gpu"
python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# Official JAX CUDA wheels (Linux only). Prefer cuda12; try cuda13 if needed.
python -m pip install --upgrade "jax[cuda12]" || python -m pip install --upgrade "jax[cuda13]"
python -m pip install --upgrade optax
cd "${REPO_ROOT}"
python -m pip install -e ".[dev]" || python -m pip install -e .

python - <<'PY'
import json, platform
from pathlib import Path
import jax
info = {
  "python": platform.python_version(),
  "jax": jax.__version__,
  "backend": jax.default_backend(),
  "devices": [str(d) for d in jax.devices()],
}
Path("experiments/manifests").mkdir(parents=True, exist_ok=True)
# pip freeze
import subprocess
freeze = subprocess.check_output(["pip", "freeze"], text=True)
Path("experiments/manifests/competition_native_jax_wsl_environment.json").write_text(
  json.dumps({"schema_version": 1, "kind": "WSL_JAX_ENV", "jax": info, "pip_freeze": freeze.splitlines()}, indent=2) + "\n",
  encoding="utf-8",
)
print(json.dumps(info))
PY
