#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
# Ensure official engine present (no-deps if already constrained)
python -m pip install -e "third_party/generals-bots" --no-deps 2>/dev/null || python -m pip install -e "third_party/generals-bots" || true
# Prefer smaller allocator fraction on 8GB laptop GPU
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.7}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
bash "${REPO_ROOT}/scripts/wsl/run_quantsilico_resume_r_e.sh" "${REPO_ROOT}"
