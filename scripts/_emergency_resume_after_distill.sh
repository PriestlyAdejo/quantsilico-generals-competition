#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition
source ~/.venvs/quantsilico-jax-gpu/bin/activate
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export EMERGENCY_RESUME_PARENT=/home/pries/quantsilico-runtime/emergency_rolling_v1/training/checkpoints/ckpt_final
rm -f /home/pries/quantsilico-runtime/emergency_rolling_v1/training/STOP_REQUEST
rm -f /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition/experiments/competition_native_jax/emergency_rolling_v1/STOP_REQUEST
exec python -u scripts/emergency_exact_resume_ppo.py
