#!/usr/bin/env bash
# STAGE5_SCALE_R2 round orchestrator (run ON the pod via
# remote_orchestrator_with_stop.sh).
# Predeclared plan: experiments/marathon/stage5_scale_r2_plan.yaml
# Registry: experiment#stage5-scale-r2 (successor to EV-0061 SCALE-R1).
# 8M win-emergence replication (seeds 20260915/20260917) + 16M depth probe.
# PPO_SEMANTICS: UNCHANGED.
set -u

REPO=/workspace/repo
PY=/workspace/repo/.venv312/bin/python
ROUND_LOG=/workspace/stage5_scale_r2_round.log
OUT_ROOT=/workspace/screening_runs_stage5_scale_r2
CKPT=/workspace/ckpt_baseline_v0
export PYTHONPATH=/workspace/repo:/workspace/repo/src:/workspace/repo/third_party/generals-bots
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "${REPO}"
mkdir -p "$OUT_ROOT"
echo "=== STAGE5 SCALE R2 round start $(stamp) commit $(cd $REPO && git rev-parse --short HEAD)" >> "$ROUND_LOG"
echo "=== SCALE-R2 arm table (predeclared): 8M x seeds {20260915 20260917} + 16M x seed 20260915, 256x32, persistent, reward none" >> "$ROUND_LOG"

for spec in \
  "SCALE-B0-8M-S3 20260915 8388608" \
  "SCALE-B0-8M-S4 20260917 8388608" \
  "SCALE-B1-16M-S1 20260915 16777216"
do
  set -- $spec
  arm=$1; seed=$2; budget=$3
  echo "=== ${arm} (256x32 seed=${seed} carry=persistent budget=${budget}) start $(stamp)" >> "$ROUND_LOG"
  timeout 3600 "$PY" scripts/training/run_sh_r1_arm.py \
    --arm-id "$arm" \
    --num-envs 256 \
    --rollout-len 32 \
    --seed "$seed" \
    --reward-shape none \
    --episode-carry persistent \
    --budget-transitions "$budget" \
    --checkpoint "$CKPT" \
    --out-dir "$OUT_ROOT/$arm" >> "$ROUND_LOG" 2>&1
  code=$?
  if [ "$code" -eq 124 ]; then
    echo "=== ${arm} exit=WALL_CAP_60MIN $(stamp)" >> "$ROUND_LOG"
  else
    echo "=== ${arm} exit=${code} $(stamp)" >> "$ROUND_LOG"
  fi
done

echo "=== STAGE5 SCALE R2 round end $(stamp)" >> "$ROUND_LOG"
