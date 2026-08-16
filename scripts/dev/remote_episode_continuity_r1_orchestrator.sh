#!/usr/bin/env bash
# EPISODE_CONTINUITY_R1 round orchestrator (run ON the pod via
# remote_orchestrator_with_stop.sh).
# Predeclared plan: experiments/marathon/episode_continuity_round_1_plan.yaml
# Registry: experiment#episode-continuity-r1#d59d66093f59
# Regime comparison: EARLY_WINDOW_RESET_REGIME_V1 (--episode-carry none) vs
# PERSISTENT_EPISODE_REGIME_V1 (--episode-carry persistent). ALL arms share
# geometry 256x32, 240 updates (~2M transitions), identity reward (none),
# baseline checkpoint, competition-default boards, SAME POD. The ONLY
# systematic difference is episode_carry. PPO_SEMANTICS: UNCHANGED.
set -u

REPO=/workspace/repo
PY=/workspace/repo/.venv312/bin/python
ROUND_LOG=/workspace/episode_continuity_r1_round.log
OUT_ROOT=/workspace/screening_runs_episode_continuity_r1
CKPT=/workspace/ckpt_baseline_v0
export PYTHONPATH=/workspace/repo:/workspace/repo/src:/workspace/repo/third_party/generals-bots
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "${REPO}"   # runner script paths are relative to the repo root
mkdir -p "$OUT_ROOT"
echo "=== EPC1 round start $(stamp) commit $(cd $REPO && git rev-parse --short HEAD)" >> "$ROUND_LOG"
echo "=== EPC1 arm table (predeclared): RESET(none)/PERSIST(persistent) x seeds {20260901 20260903}, 240 updates @ 256x32, reward none" >> "$ROUND_LOG"

for spec in \
  "EPC1-A0-RESET-S1 none 20260901" \
  "EPC1-A0-RESET-S2 none 20260903" \
  "EPC1-A1-PERSIST-S1 persistent 20260901" \
  "EPC1-A1-PERSIST-S2 persistent 20260903"
do
  set -- $spec
  arm=$1; carry=$2; seed=$3
  echo "=== ${arm} (256x32 episode_carry=${carry} seed=${seed}) start $(stamp)" >> "$ROUND_LOG"
  timeout 5400 "$PY" scripts/training/run_sh_r1_arm.py \
    --arm-id "$arm" \
    --num-envs 256 \
    --rollout-len 32 \
    --seed "$seed" \
    --episode-carry "$carry" \
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

echo "=== EPC1 round end $(stamp)" >> "$ROUND_LOG"
