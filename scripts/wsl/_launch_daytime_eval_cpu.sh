#!/usr/bin/env bash
set -euo pipefail
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition
export PYTHONPATH="src:.:third_party/generals-bots"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=
mkdir -p experiments/logs/owned_jobs
exec > experiments/logs/owned_jobs/v43_eval_cpu.out.log 2>&1
echo "START $(date -u -Iseconds)"
python scripts/run_competition_native_jax_daytime_eval.py \
  --ckpt experiments/competition_native_jax/v4_3_r_e6/ckpt_final \
  --which ema \
  --phase auto
echo "END rc=$? $(date -u -Iseconds)"
