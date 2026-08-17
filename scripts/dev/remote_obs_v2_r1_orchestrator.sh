#!/usr/bin/env bash
# OBS_V2_R1 round orchestrator (run ON the pod via
# remote_orchestrator_with_stop.sh).
# Predeclared plan: experiments/marathon/obs_v2_r1_plan.yaml
# Registry: runs registered by scripts/dev/register_obs_v2_r1_runs.py.
# Observation is the ONLY axis vs the matched RC-R1 bridge controls:
# identical recipe (256 envs x 128 persistent, topadv 0.25,
# competence-spawn curriculum stages [8,17], rc1 schedules,
# accumulate-minibatches 8) with --obs-version v2 (14-plane/12-global
# objective-aware observation; DECLARED warm-start shape surgery
# patch_proj 72->126, global_proj 8->12, old rows preserved, new rows
# deterministic, opt fresh, EMA fresh from new params).
# Budget FROZEN per predeclared rule from RC-R1 measured end-to-end TPS
# (largest whole-update budget within the 90-min/arm GPU cap).
# Seeds 20260923/20260925. PPO_SEMANTICS: UNCHANGED.
set -u

REPO=/workspace/repo
PY=/workspace/repo/.venv312/bin/python
ROUND_LOG=/workspace/obs_v2_r1_round.log
OUT_ROOT=/workspace/obs_v2_r1
BASELINE=/workspace/ckpt_baseline_v0
export PYTHONPATH=/workspace/repo:/workspace/repo/src:/workspace/repo/third_party/generals-bots
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "${REPO}"
mkdir -p "$OUT_ROOT"
echo "=== OBS-V2-R1 round start $(stamp) commit $(cd $REPO && git rev-parse --short HEAD)" >> "$ROUND_LOG"
for required in "$BASELINE/raw.npz" "$BASELINE/ema.npz"; do
  if [ ! -f "$required" ]; then
    echo "=== FATAL missing checkpoint file: $required $(stamp)" >> "$ROUND_LOG"
    exit 3
  fi
done
echo "=== arm table (predeclared): 2 arms seeds {20260923 20260925}, 256x128, persistent, reward none, warm-start MARATHON_BASELINE_V0 + DECLARED OBS-V2 shape surgery, topadv 0.25, curriculum competence-spawn [8,17], schedules rc1 (fresh opt_state), accumulate-minibatches 8, obs-version v2; budget per arm FROZEN at registration from RC-R1 TPS (90-min cap)" >> "$ROUND_LOG"

failed=0
for spec in \
  "OBS-V2-R1-S1 20260923 __BUDGET__" \
  "OBS-V2-R1-S2 20260925 __BUDGET__"
do
  set -- $spec
  arm=$1; seed=$2; budget=$3
  echo "=== ${arm} (256x128 seed=${seed} carry=persistent budget=${budget} topadv=0.25 curriculum=competence-spawn schedules=rc1 accumulate=8 obs=v2) start $(stamp)" >> "$ROUND_LOG"
  timeout 5700 "$PY" scripts/training/run_sh_r1_arm.py \
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
    echo "=== ${arm} exit=WALL_CAP_95MIN $(stamp)" >> "$ROUND_LOG"
  else
    echo "=== ${arm} exit=${code} $(stamp)" >> "$ROUND_LOG"
  fi
  if [ "$code" -ne 0 ]; then
    failed=1
  fi
done

echo "=== OBS-V2-R1 round end $(stamp)" >> "$ROUND_LOG"
exit $failed
