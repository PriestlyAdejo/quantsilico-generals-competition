#!/usr/bin/env bash
# STAGE6_DISTILL_PPO_R1 STEP D1 round orchestrator (run ON the pod via
# remote_orchestrator_with_stop.sh).
# Predeclared plan: experiments/marathon/stage6_distill_ppo_r1_plan.yaml
# Registry: experiment#stage6-distill-ppo-r1#543924456945 (EV-0067 D0 gate PASS).
# PPO continuation from the distilled teacher checkpoint: 2 x 8M arms,
# seeds 20260915/20260917, canonical 256x32 persistent identity-reward.
# PPO_SEMANTICS: UNCHANGED (initialization-only difference vs SCALE 8M arms).
set -u

REPO=/workspace/repo
PY=/workspace/repo/.venv312/bin/python
ROUND_LOG=/workspace/distill_ppo_r1_d1_round.log
OUT_ROOT=/workspace/distill_ppo_r1/d1
CKPT=/workspace/distill_ppo_r1/DISTILL-S0-TRANSFORMER-BC
export PYTHONPATH=/workspace/repo:/workspace/repo/src:/workspace/repo/third_party/generals-bots
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "${REPO}"
mkdir -p "$OUT_ROOT"
echo "=== DISTILL-PPO D1 round start $(stamp) commit $(cd $REPO && git rev-parse --short HEAD)" >> "$ROUND_LOG"
echo "=== D1 arm table (predeclared): 2 x 8M seeds {20260915 20260917}, 256x32, persistent, reward none, warm-start DISTILL-S0" >> "$ROUND_LOG"

for spec in \
  "DISTILL-PPO-S1 20260915 8388608" \
  "DISTILL-PPO-S2 20260917 8388608"
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

echo "=== DISTILL-PPO D1 round end $(stamp)" >> "$ROUND_LOG"
