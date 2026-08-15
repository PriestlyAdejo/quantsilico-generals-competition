#!/usr/bin/env bash
# SH-R4-BUDGET-ESCALATION round orchestrator (run ON the pod, nohup).
# Predeclared plan: experiments/marathon/screening_round_4_plan.yaml
# Entry condition MET at SH-R3 adjudication (EV-0031): all three geometry
# families survived cross-seed; arms materialised per the predeclared rule
# (surviving geometries x seeds {20260816, 20260818}), 240 updates each.
set -u

REPO=/workspace/repo
PY=/workspace/repo/.venv312/bin/python
ROUND_LOG=/workspace/sh_r4_round.log
OUT_ROOT=/workspace/screening_runs_r4
CKPT=/workspace/ckpt_baseline_v0
export PYTHONPATH=/workspace/repo:/workspace/repo/src:/workspace/repo/third_party/generals-bots
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "${REPO}"   # runner script paths are relative to the repo root
mkdir -p "$OUT_ROOT"
echo "=== SH-R4 round start $(stamp) commit $(cd $REPO && git rev-parse --short HEAD)" >> "$ROUND_LOG"
echo "=== SH-R4 arm table (predeclared survivors x seeds): A0-CONTROL A1-HORIZON-64 A2-HORIZON-128 x {20260816 20260818}, 240 updates @ 8192 batch" >> "$ROUND_LOG"

for spec in \
  "SH-R4-A0-CONTROL-B16 256 32 20260816" \
  "SH-R4-A0-CONTROL-B18 256 32 20260818" \
  "SH-R4-A1-HORIZON-64-B16 128 64 20260816" \
  "SH-R4-A1-HORIZON-64-B18 128 64 20260818" \
  "SH-R4-A2-HORIZON-128-B16 64 128 20260816" \
  "SH-R4-A2-HORIZON-128-B18 64 128 20260818"
do
  set -- $spec
  arm=$1; envs=$2; rlen=$3; seed=$4
  echo "=== ${arm} (${envs}x${rlen} seed=${seed}) start $(stamp)" >> "$ROUND_LOG"
  timeout 5400 "$PY" scripts/training/run_sh_r1_arm.py \
    --arm-id "$arm" \
    --num-envs "$envs" \
    --rollout-len "$rlen" \
    --seed "$seed" \
    --budget-transitions 1966080 \
    --checkpoint "$CKPT" \
    --out-dir "$OUT_ROOT/$arm" >> "$ROUND_LOG" 2>&1
  code=$?
  if [ "$code" -eq 124 ]; then
    echo "=== ${arm} exit=WALL_CAP_90MIN $(stamp)" >> "$ROUND_LOG"
  else
    echo "=== ${arm} exit=${code} $(stamp)" >> "$ROUND_LOG"
  fi
done

echo "=== SH-R4 round end $(stamp)" >> "$ROUND_LOG"
