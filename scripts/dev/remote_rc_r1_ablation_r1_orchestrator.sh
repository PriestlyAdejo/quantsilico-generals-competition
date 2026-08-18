#!/usr/bin/env bash
# RC_R1_ABLATION_R1 round orchestrator (run ON the pod via
# remote_orchestrator_with_stop.sh).
# Predeclared plan: experiments/marathon/rc_r1_ablation_r1_plan.yaml
# Registry: experiment#rc-r1-ablation-r1#1045898028bb (EV-0072 branch (b)
# successor; ONE bounded round, telemetry-grade adjudication only).
#   RC-AB1-FRAG-S1: D1 only - rollout-len 128 + accumulate-minibatches 8
#     (OOM repair, single-step semantics); NO curriculum (fixed
#     min-generals-distance 17 default); topadv 1.0 identity; NO schedules;
#     persistent; 256 envs; MARATHON_BASELINE_V0 warm start; seed 20260927;
#     budget 8,388,608 transitions (256 updates).
#   RC-AB2-CURR-S1: D2 only - competence-spawn curriculum stages [8,17]
#     with the RC-R1 advancement rule (greedy-vs-legal_random >= 0.6, 64
#     games, every 32 updates); rollout-len 32; topadv 1.0; NO schedules;
#     persistent; 256 envs; same warm start; seed 20260927; same budget.
# Matched controls exist (EV-0061 SCALE arms + RC-R1 telemetry); no
# recompute. PPO_SEMANTICS: UNCHANGED. obs-version v1 default (OBS-V1).
set -u

REPO=/workspace/repo
PY=/workspace/repo/.venv312/bin/python
ROUND_LOG=/workspace/rc_r1_ablation_r1_round.log
OUT_ROOT=/workspace/rc_r1_ablation_r1
BASELINE=/workspace/ckpt_baseline_v0
export PYTHONPATH=/workspace/repo:/workspace/repo/src:/workspace/repo/third_party/generals-bots
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "${REPO}"
echo "=== RC-R1 ABLATION R1 round start $(stamp) commit $(git rev-parse --short HEAD)" >> "$ROUND_LOG"
for required in "$BASELINE/raw.npz" "$BASELINE/ema.npz" "$BASELINE/opt_state.npz"; do
  if [ ! -f "$required" ]; then
    echo "=== FATAL missing checkpoint file: $required $(stamp)" >> "$ROUND_LOG"
    exit 3
  fi
done
mkdir -p "$OUT_ROOT"
for arm in RC-AB1-FRAG-S1 RC-AB2-CURR-S1; do
  if [ -e "$OUT_ROOT/$arm" ] && [ -n "$(ls -A "$OUT_ROOT/$arm" 2>/dev/null)" ]; then
    echo "=== FATAL overwrite guard: $OUT_ROOT/$arm not clean $(stamp)" >> "$ROUND_LOG"
    exit 4
  fi
done
echo "=== arm table (predeclared): RC-AB1-FRAG-S1 D1-only 256x128 accumulate=8 topadv=1.0 no-curriculum no-schedules; RC-AB2-CURR-S1 D2-only 256x32 competence-spawn[8,17] topadv=1.0 no-schedules; both persistent seed=20260927 budget=8388608 warm-start MARATHON_BASELINE_V0" >> "$ROUND_LOG"

failed=0

echo "=== RC-AB1-FRAG-S1 start $(stamp)" >> "$ROUND_LOG"
timeout 4200 "$PY" scripts/training/run_sh_r1_arm.py \
  --arm-id RC-AB1-FRAG-S1 \
  --num-envs 256 \
  --rollout-len 128 \
  --seed 20260927 \
  --reward-shape none \
  --episode-carry persistent \
  --budget-transitions 8388608 \
  --checkpoint "$BASELINE" \
  --top-advantage-fraction 1.0 \
  --curriculum none \
  --schedules none \
  --accumulate-minibatches 8 \
  --out-dir "$OUT_ROOT/RC-AB1-FRAG-S1" >> "$ROUND_LOG" 2>&1
code=$?
if [ "$code" -eq 124 ]; then
  echo "=== RC-AB1-FRAG-S1 exit=WALL_CAP_70MIN $(stamp)" >> "$ROUND_LOG"
else
  echo "=== RC-AB1-FRAG-S1 exit=${code} $(stamp)" >> "$ROUND_LOG"
fi
[ "$code" -ne 0 ] && failed=1

echo "=== RC-AB2-CURR-S1 start $(stamp)" >> "$ROUND_LOG"
timeout 4200 "$PY" scripts/training/run_sh_r1_arm.py \
  --arm-id RC-AB2-CURR-S1 \
  --num-envs 256 \
  --rollout-len 32 \
  --seed 20260927 \
  --reward-shape none \
  --episode-carry persistent \
  --budget-transitions 8388608 \
  --checkpoint "$BASELINE" \
  --top-advantage-fraction 1.0 \
  --curriculum competence-spawn \
  --schedules none \
  --out-dir "$OUT_ROOT/RC-AB2-CURR-S1" >> "$ROUND_LOG" 2>&1
code=$?
if [ "$code" -eq 124 ]; then
  echo "=== RC-AB2-CURR-S1 exit=WALL_CAP_70MIN $(stamp)" >> "$ROUND_LOG"
else
  echo "=== RC-AB2-CURR-S1 exit=${code} $(stamp)" >> "$ROUND_LOG"
fi
[ "$code" -ne 0 ] && failed=1

echo "=== RC-R1 ABLATION R1 round end $(stamp)" >> "$ROUND_LOG"
exit $failed
