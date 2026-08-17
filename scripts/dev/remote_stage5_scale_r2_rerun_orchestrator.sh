#!/usr/bin/env bash
# STAGE5_SCALE_R2 arm-1 RERUN orchestrator (bounded engineering repair).
# EV record pending (EV-0062): SCALE-B0-8M-S3 completed its 20-min training
# but died at the final telemetry flush with OSError Errno 5 (transient
# network-volume I/O error on mfs eu-se-1.runpod.net); no summary/checkpoint
# written. Repair = rerun the IDENTICAL predeclared config (seed 20260915,
# 8M transitions) - recovery of a failed hardware execution, not a new
# experiment and not a rule change.
# PPO_SEMANTICS: UNCHANGED.
set -u

REPO=/workspace/repo
PY=/workspace/repo/.venv312/bin/python
ROUND_LOG=/workspace/stage5_scale_r2_rerun_round.log
OUT_ROOT=/workspace/screening_runs_stage5_scale_r2
CKPT=/workspace/ckpt_baseline_v0
export PYTHONPATH=/workspace/repo:/workspace/repo/src:/workspace/repo/third_party/generals-bots
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "${REPO}"
mkdir -p "$OUT_ROOT"
echo "=== STAGE5 SCALE R2 arm-1 RERUN start $(stamp) commit $(cd $REPO && git rev-parse --short HEAD) (repair for OSError-5 flush failure 00:38:32Z)" >> "$ROUND_LOG"
echo "=== SCALE-B0-8M-S3-RERUN (256x32 seed=20260915 carry=persistent budget=8388608) start $(stamp)" >> "$ROUND_LOG"
timeout 3600 "$PY" scripts/training/run_sh_r1_arm.py \
  --arm-id SCALE-B0-8M-S3 \
  --num-envs 256 \
  --rollout-len 32 \
  --seed 20260915 \
  --reward-shape none \
  --episode-carry persistent \
  --budget-transitions 8388608 \
  --checkpoint "$CKPT" \
  --out-dir "$OUT_ROOT/SCALE-B0-8M-S3" >> "$ROUND_LOG" 2>&1
code=$?
if [ "$code" -eq 124 ]; then
  echo "=== SCALE-B0-8M-S3-RERUN exit=WALL_CAP_60MIN $(stamp)" >> "$ROUND_LOG"
else
  echo "=== SCALE-B0-8M-S3-RERUN exit=${code} $(stamp)" >> "$ROUND_LOG"
fi
echo "=== STAGE5 SCALE R2 arm-1 RERUN end $(stamp)" >> "$ROUND_LOG"
