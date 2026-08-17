#!/usr/bin/env bash
# STAGE6_OPPDIST_R1 round orchestrator (run ON the pod via
# remote_orchestrator_with_stop.sh).
# Predeclared plan: experiments/marathon/stage6_oppdist_r1_plan.yaml
# Registry: experiment#stage6-oppdist-r1#c0103074d017, runs
# run#oppdist-b0-teachopp-s1#b956cd1a9bea / run#oppdist-b0-teachopp-s2#20ac28eb3eb5.
# 2 x 8M arms, seeds 20260915/20260917, canonical 256x32 persistent
# identity-reward, warm start MARATHON_BASELINE_V0 (matched to SCALE controls),
# seat 1 = FROZEN DISTILL-S0 teacher checkpoint (loaded once, never updated,
# sampled stochastically temperature 1.0; PPO on seat-0 samples only).
# PPO_SEMANTICS: UNCHANGED (opponent is environment dynamics).
set -u

REPO=/workspace/repo
PY=/workspace/repo/.venv312/bin/python
ROUND_LOG=/workspace/oppdist_r1_round.log
OUT_ROOT=/workspace/oppdist_r1
BASELINE=/workspace/ckpt_baseline_v0
OPPONENT=/workspace/distill_ppo_r1/DISTILL-S0-TRANSFORMER-BC
export PYTHONPATH=/workspace/repo:/workspace/repo/src:/workspace/repo/third_party/generals-bots
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "${REPO}"
mkdir -p "$OUT_ROOT"
echo "=== OPPDIST R1 round start $(stamp) commit $(cd $REPO && git rev-parse --short HEAD)" >> "$ROUND_LOG"
for required in "$BASELINE/raw.npz" "$BASELINE/ema.npz" "$BASELINE/opt_state.npz" "$OPPONENT/raw.npz"; do
  if [ ! -f "$required" ]; then
    echo "=== FATAL missing checkpoint file: $required $(stamp)" >> "$ROUND_LOG"
    exit 3
  fi
done
echo "=== R1 arm table (predeclared): 2 x 8M seeds {20260915 20260917}, 256x32, persistent, reward none, warm-start MARATHON_BASELINE_V0, opponent FROZEN DISTILL-S0 temp 1.0" >> "$ROUND_LOG"

for spec in \
  "OPPDIST-B0-TEACHOPP-S1 20260915 8388608" \
  "OPPDIST-B0-TEACHOPP-S2 20260917 8388608"
do
  set -- $spec
  arm=$1; seed=$2; budget=$3
  echo "=== ${arm} (256x32 seed=${seed} carry=persistent budget=${budget} opponent=teacher_frozen) start $(stamp)" >> "$ROUND_LOG"
  timeout 3600 "$PY" scripts/training/run_sh_r1_arm.py \
    --arm-id "$arm" \
    --num-envs 256 \
    --rollout-len 32 \
    --seed "$seed" \
    --reward-shape none \
    --episode-carry persistent \
    --budget-transitions "$budget" \
    --checkpoint "$BASELINE" \
    --opponent-mode teacher_frozen \
    --opponent-checkpoint "$OPPONENT" \
    --out-dir "$OUT_ROOT/$arm" >> "$ROUND_LOG" 2>&1
  code=$?
  if [ "$code" -eq 124 ]; then
    echo "=== ${arm} exit=WALL_CAP_60MIN $(stamp)" >> "$ROUND_LOG"
  else
    echo "=== ${arm} exit=${code} $(stamp)" >> "$ROUND_LOG"
  fi
done

echo "=== OPPDIST R1 round end $(stamp)" >> "$ROUND_LOG"
