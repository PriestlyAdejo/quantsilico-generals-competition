#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
python "${REPO_ROOT}/scripts/wsl/verify_quantsilico_jax_gpu.py" "${REPO_ROOT}"
