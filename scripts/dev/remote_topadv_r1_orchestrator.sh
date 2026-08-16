#!/usr/bin/env bash
# TOPADV-R1 round orchestrator (run ON the pod via remote_orchestrator_with_stop.sh).
# Predeclared plan: experiments/marathon/topadv_round_1_plan.yaml
# All arms share the SH-R4 finalist geometry (256x32, 240 updates, ~2M
# transitions) and competition-default boards; the ONLY systematic difference
# is top_advantage_fraction. PPO_SEMANTICS: UNCHANGED (serving untouched).
set -u

REPO=/workspace/repo
PY=/workspace/repo/.venv312/bin/python
ROUND_LOG=/workspace/topadv_r1_round.log
OUT_ROOT=/workspace/screening_runs_topadv_r1
CKPT=/workspace/ckpt_baseline_v0
export PYTHONPATH=/workspace/repo:/workspace/repo/src:/workspace/repo/third_party/generals-bots
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "${REPO}"   # runner script paths are relative to the repo root
mkdir -p "$OUT_ROOT"
echo "=== TOPADV-R1 round start $(stamp) commit $(cd $REPO && git rev-parse --short HEAD)" >> "$ROUND_LOG"
echo "=== TOPADV-R1 arm table (predeclared): CONTROL(1.0)/TOP50(0.5)/TOP25(0.25) x seeds {20260821 20260823}, 240 updates @ 256x32" >> "$ROUND_LOG"

for spec in \
  "TOPADV-A0-CONTROL-S1 1.0 20260821" \
  "TOPADV-A0-CONTROL-S2 1.0 20260823" \
  "TOPADV-A1-TOP50-S1 0.5 20260821" \
  "TOPADV-A1-TOP50-S2 0.5 20260823" \
  "TOPADV-A2-TOP25-S1 0.25 20260821" \
  "TOPADV-A2-TOP25-S2 0.25 20260823"
do
  set -- $spec
  arm=$1; frac=$2; seed=$3
  echo "=== ${arm} (256x32 frac=${frac} seed=${seed}) start $(stamp)" >> "$ROUND_LOG"
  timeout 5400 "$PY" scripts/training/run_sh_r1_arm.py \
    --arm-id "$arm" \
    --num-envs 256 \
    --rollout-len 32 \
    --seed "$seed" \
    --top-advantage-fraction "$frac" \
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

echo "=== TOPADV-R1 round end $(stamp)" >> "$ROUND_LOG"
