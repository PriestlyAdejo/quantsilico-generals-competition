#!/usr/bin/env bash
# SH-R3-SEEDS round orchestrator (run ON the pod, nohup).
# Predeclared plan: experiments/marathon/screening_round_3_plan.yaml
# Arms run SEQUENTIALLY (each ~39 GiB VRAM on A40); pool-fixed runner.
set -u

REPO=/workspace/repo
PY=/workspace/repo/.venv312/bin/python
ROUND_LOG=/workspace/sh_r3_round.log
OUT_ROOT=/workspace/screening_runs_r3
CKPT=/workspace/ckpt_baseline_v0
export PYTHONPATH=/workspace/repo:/workspace/repo/src:/workspace/repo/third_party/generals-bots
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p "$OUT_ROOT"
echo "=== SH-R3 round start $(stamp) commit $(cd $REPO && git rev-parse --short HEAD)" >> "$ROUND_LOG"

# arm_id num_envs rollout_len seed  (geometries x seeds per plan)
for spec in \
  "SH-R3-A0-CONTROL-S1 256 32 20260816" \
  "SH-R3-A0-CONTROL-S2 256 32 20260817" \
  "SH-R3-A1-HORIZON-64-S1 128 64 20260816" \
  "SH-R3-A1-HORIZON-64-S2 128 64 20260817" \
  "SH-R3-A2-HORIZON-128-S1 64 128 20260816" \
  "SH-R3-A2-HORIZON-128-S2 64 128 20260817"
do
  set -- $spec
  arm=$1; envs=$2; rlen=$3; seed=$4
  echo "=== ${arm} (${envs}x${rlen} seed=${seed}) start $(stamp)" >> "$ROUND_LOG"
  "$PY" scripts/training/run_sh_r1_arm.py \
    --arm-id "$arm" \
    --num-envs "$envs" \
    --rollout-len "$rlen" \
    --seed "$seed" \
    --budget-transitions 491520 \
    --checkpoint "$CKPT" \
    --out-dir "$OUT_ROOT/$arm" >> "$ROUND_LOG" 2>&1
  code=$?
  echo "=== ${arm} exit=${code} $(stamp)" >> "$ROUND_LOG"
done

echo "=== SH-R3 round end $(stamp)" >> "$ROUND_LOG"
