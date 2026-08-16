#!/usr/bin/env bash
# STAGE5_SCALE_R1 round orchestrator (run ON the pod via
# remote_orchestrator_with_stop.sh).
# Predeclared plan: experiments/marathon/stage5_scale_r1_plan.yaml
# Registry: experiment#stage5-scale-r1#600fdb29e986
# Runs REGISTERED pre-launch: run#scale-a0-8m-s1#62285c0c59e1 (seed 20260911),
# run#scale-a0-8m-s2#e7ec639079d9 (seed 20260913).
# Question: is transition scale the binding constraint on win conversion?
# Canonical lineage, persistent regime, identity reward, canonical geometry,
# 8M transitions per seed; gameplay arbiter is the ONLY gate.
# PPO_SEMANTICS: UNCHANGED.
set -u

REPO=/workspace/repo
PY=/workspace/repo/.venv312/bin/python
ROUND_LOG=/workspace/stage5_scale_r1_round.log
OUT_ROOT=/workspace/screening_runs_stage5_scale_r1
CKPT=/workspace/ckpt_baseline_v0
export PYTHONPATH=/workspace/repo:/workspace/repo/src:/workspace/repo/third_party/generals-bots
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "${REPO}"
mkdir -p "$OUT_ROOT"
echo "=== STAGE5 SCALE R1 round start $(stamp) commit $(cd $REPO && git rev-parse --short HEAD)" >> "$ROUND_LOG"
echo "=== SCALE arm table (predeclared): 8M transitions x seeds {20260911 20260913}, 256x32, episode_carry=persistent, reward none" >> "$ROUND_LOG"

for spec in \
  "SCALE-A0-8M-S1 20260911" \
  "SCALE-A0-8M-S2 20260913"
do
  set -- $spec
  arm=$1; seed=$2
  echo "=== ${arm} (256x32 seed=${seed} carry=persistent budget=8388608) start $(stamp)" >> "$ROUND_LOG"
  timeout 3600 "$PY" scripts/training/run_sh_r1_arm.py \
    --arm-id "$arm" \
    --num-envs 256 \
    --rollout-len 32 \
    --seed "$seed" \
    --reward-shape none \
    --episode-carry persistent \
    --budget-transitions 8388608 \
    --checkpoint "$CKPT" \
    --out-dir "$OUT_ROOT/$arm" >> "$ROUND_LOG" 2>&1
  code=$?
  if [ "$code" -eq 124 ]; then
    echo "=== ${arm} exit=WALL_CAP_60MIN $(stamp)" >> "$ROUND_LOG"
  else
    echo "=== ${arm} exit=${code} $(stamp)" >> "$ROUND_LOG"
  fi
done

echo "=== STAGE5 SCALE R1 round end $(stamp)" >> "$ROUND_LOG"
