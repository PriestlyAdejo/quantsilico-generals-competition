#!/usr/bin/env bash
# SPAWN-DISTANCE-CURRICULUM-R1 round orchestrator (run ON the pod, nohup).
# Predeclared plan: experiments/marathon/curriculum_round_1_plan.yaml
# All arms share the SH-R4 finalist geometry (256x32, 240 updates, ~2M
# transitions); the ONLY systematic difference is min_generals_distance.
# PPO_SEMANTICS: UNCHANGED (map generation only). EV-0035 entry condition MET.
set -u

REPO=/workspace/repo
PY=/workspace/repo/.venv312/bin/python
ROUND_LOG=/workspace/curr1_round.log
OUT_ROOT=/workspace/screening_runs_curr1
CKPT=/workspace/ckpt_baseline_v0
export PYTHONPATH=/workspace/repo:/workspace/repo/src:/workspace/repo/third_party/generals-bots
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "${REPO}"   # runner script paths are relative to the repo root
mkdir -p "$OUT_ROOT"
echo "=== CURR1 round start $(stamp) commit $(cd $REPO && git rev-parse --short HEAD)" >> "$ROUND_LOG"
echo "=== CURR1 arm table (predeclared): CONTROL(17)/CLOSE(8)/FAR(21) x seeds {20260816 20260818}, 240 updates @ 256x32" >> "$ROUND_LOG"

for spec in \
  "CURR1-A0-CONTROL-S1 17 20260816" \
  "CURR1-A0-CONTROL-S2 17 20260818" \
  "CURR1-A1-CLOSE-8-S1 8 20260816" \
  "CURR1-A1-CLOSE-8-S2 8 20260818" \
  "CURR1-A2-FAR-21-S1 21 20260816" \
  "CURR1-A2-FAR-21-S2 21 20260818"
do
  set -- $spec
  arm=$1; dist=$2; seed=$3
  echo "=== ${arm} (256x32 dist=${dist} seed=${seed}) start $(stamp)" >> "$ROUND_LOG"
  timeout 5400 "$PY" scripts/training/run_sh_r1_arm.py \
    --arm-id "$arm" \
    --num-envs 256 \
    --rollout-len 32 \
    --seed "$seed" \
    --min-generals-distance "$dist" \
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

echo "=== CURR1 round end $(stamp)" >> "$ROUND_LOG"
