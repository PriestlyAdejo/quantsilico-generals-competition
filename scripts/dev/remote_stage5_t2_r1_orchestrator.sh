#!/usr/bin/env bash
# STAGE5_CAPACITY_VALUE_R1 T2 round orchestrator (run ON the pod via
# remote_orchestrator_with_stop.sh).
# Predeclared plan: experiments/marathon/stage5_capacity_value_r1_plan.yaml
# Registry: experiment#stage5-capacity-value-r1#f1dc01ccc86f
# Runs REGISTERED pre-launch: run#s5-t2-k1-s1#47d1e62130d8 (seed 20260905),
# run#s5-t2-k1-s2#6eac713a3ba6 (seed 20260907).
# Fallback trigger (predeclared): local CPU insufficient at canonical
# geometry - 256x32 OOM on 15.8GB box (contention + solo probe); bounded A40
# round per plan budget clause. ONLY systematic difference vs T0 (RWB1
# controls): --temporal-history k1 (prev-tick legal spatial planes; shared
# layers warm-started from baseline, patch_proj shape-forced fresh).
# PPO_SEMANTICS: UNCHANGED.
set -u

REPO=/workspace/repo
PY=/workspace/repo/.venv312/bin/python
ROUND_LOG=/workspace/stage5_t2_r1_round.log
OUT_ROOT=/workspace/screening_runs_stage5_t2_r1
CKPT=/workspace/ckpt_baseline_v0
export PYTHONPATH=/workspace/repo:/workspace/repo/src:/workspace/repo/third_party/generals-bots
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "${REPO}"
mkdir -p "$OUT_ROOT"
echo "=== STAGE5 T2 round start $(stamp) commit $(cd $REPO && git rev-parse --short HEAD)" >> "$ROUND_LOG"
echo "=== T2 arm table (predeclared): k1 temporal history x seeds {20260905 20260907}, 240 updates @ 256x32, episode_carry=persistent, reward none" >> "$ROUND_LOG"

for spec in \
  "S5-T2-K1-S1 20260905" \
  "S5-T2-K1-S2 20260907"
do
  set -- $spec
  arm=$1; seed=$2
  echo "=== ${arm} (256x32 k1 seed=${seed} carry=persistent) start $(stamp)" >> "$ROUND_LOG"
  timeout 5400 "$PY" scripts/training/run_sh_r1_arm.py \
    --arm-id "$arm" \
    --num-envs 256 \
    --rollout-len 32 \
    --seed "$seed" \
    --temporal-history k1 \
    --reward-shape none \
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

echo "=== STAGE5 T2 round end $(stamp)" >> "$ROUND_LOG"
