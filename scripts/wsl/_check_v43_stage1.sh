#!/usr/bin/env bash
set -euo pipefail
LOG=/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition/experiments/logs/owned_jobs/v43_stage3a3b_20260806_180027.out.log
echo "=== tail ==="
tail -n 80 "$LOG"
echo "=== artefacts ==="
ls -la /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition/experiments/manifests/competition_native_jax_v4_2_stage3*.json 2>/dev/null || true
echo "=== processes ==="
pgrep -af '_stage1_v4_3|_run_v4_3_stage3a|train_jax' || echo IDLE
