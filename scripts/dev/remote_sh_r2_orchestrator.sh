#!/usr/bin/env bash
# SH-R2 sequential round orchestrator (run ON the A40 pod).
# Runs the three predeclared arms back-to-back (each ~39 GiB VRAM, so never
# concurrently), appending arm boundaries + runner output to sh_r2_round.log.
set -uo pipefail

cd /workspace/repo
export PYTHONPATH=/workspace/repo:/workspace/repo/src:/workspace/repo/third_party/generals-bots

ROUND_LOG=/workspace/sh_r2_round.log
echo "=== SH-R2 round start $(date -u +%FT%TZ) commit $(git rev-parse --short HEAD)" >> "$ROUND_LOG"

for spec in "SH-R2-A0-CONTROL 256 32" "SH-R2-A1-HORIZON-64 128 64" "SH-R2-A2-HORIZON-128 64 128"; do
  set -- $spec
  arm=$1; envs=$2; rlen=$3
  echo "=== $arm (${envs}x${rlen}) start $(date -u +%FT%TZ)" >> "$ROUND_LOG"
  /workspace/repo/.venv312/bin/python scripts/training/run_sh_r1_arm.py \
    --arm-id "$arm" --num-envs "$envs" --rollout-len "$rlen" \
    --budget-transitions 491520 \
    --checkpoint /workspace/ckpt_baseline_v0 \
    --out-dir /workspace/screening_runs/"$arm" >> "$ROUND_LOG" 2>&1
  echo "=== $arm exit=$? $(date -u +%FT%TZ)" >> "$ROUND_LOG"
done

echo "=== SH-R2 round end $(date -u +%FT%TZ)" >> "$ROUND_LOG"
