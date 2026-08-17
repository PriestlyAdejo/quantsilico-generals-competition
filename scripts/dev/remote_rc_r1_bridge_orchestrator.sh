#!/usr/bin/env bash
# RC_R1_BRIDGE round orchestrator (run ON the pod via
# remote_orchestrator_with_stop.sh).
# Predeclared plan: experiments/marathon/rc_r1_bridge_plan.yaml
# Registry: experiment#rc-r1-bridge#60729d9ef92f, runs
# run#rc-r1-bridge-s1#0374c1939762 / run#rc-r1-bridge-s2#6c705ce0a7bf.
# Exactly four frozen deltas vs matched SCALE controls (EV-0061):
#   D1 fragment 32 -> 128; D2 competence-spawn curriculum stages [8,17];
#   D3 top-advantage 0.25; D4 rc1 entropy/LR schedules (fresh opt_state).
# 2 x 8M arms, seeds 20260919/20260921, 256 envs, persistent, identity
# reward, warm start MARATHON_BASELINE_V0. PPO_SEMANTICS: UNCHANGED.
set -u

REPO=/workspace/repo
PY=/workspace/repo/.venv312/bin/python
ROUND_LOG=/workspace/rc_r1_bridge_round.log
OUT_ROOT=/workspace/rc_r1_bridge
BASELINE=/workspace/ckpt_baseline_v0
export PYTHONPATH=/workspace/repo:/workspace/repo/src:/workspace/repo/third_party/generals-bots
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "${REPO}"
mkdir -p "$OUT_ROOT"
echo "=== RC-R1 BRIDGE round start $(stamp) commit $(cd $REPO && git rev-parse --short HEAD)" >> "$ROUND_LOG"
for required in "$BASELINE/raw.npz" "$BASELINE/ema.npz"; do
  if [ ! -f "$required" ]; then
    echo "=== FATAL missing checkpoint file: $required $(stamp)" >> "$ROUND_LOG"
    exit 3
  fi
done
echo "=== arm table (predeclared): 2 x 8M seeds {20260919 20260921}, 256x128, persistent, reward none, warm-start MARATHON_BASELINE_V0, topadv 0.25, curriculum competence-spawn [8,17], schedules rc1 (fresh opt_state), accumulate-minibatches 8 (OOM repair, single-step semantics)" >> "$ROUND_LOG"

failed=0
for spec in \
  "RC-R1-BRIDGE-S1 20260919 8388608" \
  "RC-R1-BRIDGE-S2 20260921 8388608"
do
  set -- $spec
  arm=$1; seed=$2; budget=$3
  echo "=== ${arm} (256x128 seed=${seed} carry=persistent budget=${budget} topadv=0.25 curriculum=competence-spawn schedules=rc1 accumulate=8) start $(stamp)" >> "$ROUND_LOG"
  timeout 3600 "$PY" scripts/training/run_sh_r1_arm.py \
    --arm-id "$arm" \
    --num-envs 256 \
    --rollout-len 128 \
    --seed "$seed" \
    --reward-shape none \
    --episode-carry persistent \
    --budget-transitions "$budget" \
    --checkpoint "$BASELINE" \
    --top-advantage-fraction 0.25 \
    --curriculum competence-spawn \
    --schedules rc1 \
    --accumulate-minibatches 8 \
    --out-dir "$OUT_ROOT/$arm" >> "$ROUND_LOG" 2>&1
  code=$?
  if [ "$code" -eq 124 ]; then
    echo "=== ${arm} exit=WALL_CAP_60MIN $(stamp)" >> "$ROUND_LOG"
  else
    echo "=== ${arm} exit=${code} $(stamp)" >> "$ROUND_LOG"
  fi
  if [ "$code" -ne 0 ]; then
    failed=1
  fi
done

echo "=== RC-R1 BRIDGE round end $(stamp)" >> "$ROUND_LOG"
exit $failed
