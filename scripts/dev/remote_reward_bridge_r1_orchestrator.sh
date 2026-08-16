#!/usr/bin/env bash
# REWARD-BRIDGE-R1 round orchestrator (run ON the pod via
# remote_orchestrator_with_stop.sh).
# Predeclared plan: experiments/marathon/reward_bridge_round_1_plan.yaml
# Registry: experiment#reward-bridge-r1#a2b8ec6f32b8
# CROSS_REGIME_BRIDGE_V1 step B2: reward family reopened under the canonical
# PERSISTENT_EPISODE_REGIME_V1 (--episode-carry persistent for ALL arms;
# runner default since EV-0049). The ONLY systematic differences are the
# reward-shaping knobs. PPO_SEMANTICS: UNCHANGED (training-reward ablation).
set -u

REPO=/workspace/repo
PY=/workspace/repo/.venv312/bin/python
ROUND_LOG=/workspace/reward_bridge_r1_round.log
OUT_ROOT=/workspace/screening_runs_reward_bridge_r1
CKPT=/workspace/ckpt_baseline_v0
export PYTHONPATH=/workspace/repo:/workspace/repo/src:/workspace/repo/third_party/generals-bots
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "${REPO}"   # runner script paths are relative to the repo root
mkdir -p "$OUT_ROOT"
echo "=== RWB1 round start $(stamp) commit $(cd $REPO && git rev-parse --short HEAD)" >> "$ROUND_LOG"
echo "=== RWB1 arm table (predeclared): CONTROL(none)/LAND(1.0)/KILL(0.01) x seeds {20260901 20260903}, 240 updates @ 256x32, episode_carry=persistent" >> "$ROUND_LOG"

for spec in \
  "RWB1-A0-CONTROL-S1 none 0.0 20260901" \
  "RWB1-A0-CONTROL-S2 none 0.0 20260903" \
  "RWB1-A1-LAND-S1 land_potential 1.0 20260901" \
  "RWB1-A1-LAND-S2 land_potential 1.0 20260903" \
  "RWB1-A2-KILL-S1 kill_delta 0.01 20260901" \
  "RWB1-A2-KILL-S2 kill_delta 0.01 20260903"
do
  set -- $spec
  arm=$1; mode=$2; beta=$3; seed=$4
  echo "=== ${arm} (256x32 mode=${mode} beta=${beta} seed=${seed} carry=persistent) start $(stamp)" >> "$ROUND_LOG"
  timeout 5400 "$PY" scripts/training/run_sh_r1_arm.py \
    --arm-id "$arm" \
    --num-envs 256 \
    --rollout-len 32 \
    --seed "$seed" \
    --reward-shape "$mode" \
    --reward-shape-beta "$beta" \
    --episode-carry persistent \
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

echo "=== RWB1 round end $(stamp)" >> "$ROUND_LOG"
