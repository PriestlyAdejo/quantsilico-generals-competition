#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${REPO_ROOT}/third_party/generals-bots"
export PYTHONUNBUFFERED=1
# Daytime eval is NumPy/CPU deployment path — do not consume the training GPU.
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG="experiments/logs/owned_jobs/v43_daytime_eval_${STAMP}.out.log"
mkdir -p experiments/logs/owned_jobs
exec > >(tee -a "${LOG}") 2>&1

CKPT="${1:-experiments/competition_native_jax/v4_3_r_e6/ckpt_final}"
echo "=== CNJ daytime eval ${STAMP} ckpt=${CKPT} (CPU) ==="
python scripts/run_competition_native_jax_daytime_eval.py --ckpt "${CKPT}" --which ema --phase auto
