#!/usr/bin/env bash
# OBS_V2_R1 S2 RERUN orchestrator - bounded engineering repair (EV-0073).
# Run ON the pod via remote_orchestrator_with_stop.sh.
# Incident: the round-1 95-min per-arm backstop (timeout 5700) was too
# tight for the FROZEN 568-update budget: S1 finished ~seconds inside it
# (10.04 s/update), S2 at 10.14 s/update was killed at 562/568 with no
# checkpoint (telemetry preserved as telemetry_truncated_ev0073.jsonl).
# Repair: rerun arm OBS-V2-R1-S2 with IDENTICAL config and seed; the ONLY
# change is the protective backstop 5700 -> 6300 s (105 min). The frozen
# budget (18,612,224 transitions), all scientific knobs, and adjudication
# rules are UNTOUCHED; the trainer still self-terminates at 568 updates.
set -u

REPO=/workspace/repo
PY=/workspace/repo/.venv312/bin/python
ROUND_LOG=/workspace/obs_v2_r1_s2_rerun_round.log
OUT_ROOT=/workspace/obs_v2_r1
BASELINE=/workspace/ckpt_baseline_v0
export PYTHONPATH=/workspace/repo:/workspace/repo/src:/workspace/repo/third_party/generals-bots
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "${REPO}"
mkdir -p "$OUT_ROOT"
echo "=== OBS-V2-R1 S2 RERUN start $(stamp) commit $(cd $REPO && git rev-parse --short HEAD) repair=EV-0073 (backstop 5700->6300; config + seed + frozen budget IDENTICAL)" >> "$ROUND_LOG"
for required in "$BASELINE/raw.npz" "$BASELINE/ema.npz"; do
  if [ ! -f "$required" ]; then
    echo "=== FATAL missing checkpoint file: $required $(stamp)" >> "$ROUND_LOG"
    exit 3
  fi
done
if [ -e "$OUT_ROOT/OBS-V2-R1-S2/telemetry.jsonl" ]; then
  echo "=== FATAL overwrite guard: S2 out-dir not clean $(stamp)" >> "$ROUND_LOG"
  exit 4
fi

arm=OBS-V2-R1-S2; seed=20260925; budget=18612224
echo "=== ${arm} (256x128 seed=${seed} carry=persistent budget=${budget} topadv=0.25 curriculum=competence-spawn schedules=rc1 accumulate=8 obs=v2) start $(stamp)" >> "$ROUND_LOG"
timeout 6300 "$PY" scripts/training/run_sh_r1_arm.py \
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
  --obs-version v2 \
  --out-dir "$OUT_ROOT/$arm" >> "$ROUND_LOG" 2>&1
code=$?
if [ "$code" -eq 124 ]; then
  echo "=== ${arm} exit=WALL_CAP_105MIN $(stamp)" >> "$ROUND_LOG"
else
  echo "=== ${arm} exit=${code} $(stamp)" >> "$ROUND_LOG"
fi
echo "=== OBS-V2-R1 S2 RERUN end $(stamp)" >> "$ROUND_LOG"
exit $code
