#!/usr/bin/env bash
# REWARD-SHAPING-R1 round orchestrator (run ON the pod via remote_orchestrator_with_stop.sh).
# Predeclared plan: experiments/marathon/reward_shaping_round_1_plan.yaml
# All arms share the matched geometry (256x32, 240 updates, ~2M transitions)
# and competition-default boards; the ONLY systematic difference is the
# training-reward shaping mode/beta. PPO_SEMANTICS: UNCHANGED (serving untouched).
set -u

REPO=/workspace/repo
PY=/workspace/repo/.venv312/bin/python
ROUND_LOG=/workspace/reward_shaping_r1_round.log
OUT_ROOT=/workspace/screening_runs_reward_shaping_r1
CKPT=/workspace/ckpt_baseline_v0
export PYTHONPATH=/workspace/repo:/workspace/repo/src:/workspace/repo/third_party/generals-bots
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "${REPO}"   # runner script paths are relative to the repo root
mkdir -p "$OUT_ROOT"
echo "=== REWARD-SHAPING-R1 round start $(stamp) commit $(cd $REPO && git rev-parse --short HEAD)" >> "$ROUND_LOG"
echo "=== REWARD-SHAPING-R1 arm table (predeclared): CONTROL(none)/KILL-DELTA(0.01)/POTENTIAL(1.0) x seeds {20260825 20260827}, 240 updates @ 256x32" >> "$ROUND_LOG"

for spec in \
  "RSH-A0-CONTROL-S1 none 0.0 20260825" \
  "RSH-A0-CONTROL-S2 none 0.0 20260827" \
  "RSH-A1-KILL-DELTA-S1 kill_delta 0.01 20260825" \
  "RSH-A1-KILL-DELTA-S2 kill_delta 0.01 20260827" \
  "RSH-A2-POTENTIAL-S1 potential 1.0 20260825" \
  "RSH-A2-POTENTIAL-S2 potential 1.0 20260827"
do
  set -- $spec
  arm=$1; mode=$2; beta=$3; seed=$4
  echo "=== ${arm} (256x32 mode=${mode} beta=${beta} seed=${seed}) start $(stamp)" >> "$ROUND_LOG"
  timeout 5400 "$PY" scripts/training/run_sh_r1_arm.py \
    --arm-id "$arm" \
    --num-envs 256 \
    --rollout-len 32 \
    --seed "$seed" \
    --reward-shape "$mode" \
    --reward-shape-beta "$beta" \
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

echo "=== REWARD-SHAPING-R1 round end $(stamp)" >> "$ROUND_LOG"
