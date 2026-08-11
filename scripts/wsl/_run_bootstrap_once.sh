#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
exec bash "${REPO_ROOT}/scripts/wsl/bootstrap_quantsilico_jax_gpu.sh" "${REPO_ROOT}"
