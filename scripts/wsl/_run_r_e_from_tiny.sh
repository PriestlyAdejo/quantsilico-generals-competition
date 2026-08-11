#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export PYTHONUNBUFFERED=1

echo "=== R-E.3 tiny ==="
python -u -m train.competition_native_jax.train_jax --mode tiny --out experiments/competition_native_jax/tiny --num-envs 2 --rollout-len 8

echo "=== R-E.4 throughput ==="
python -u -m train.competition_native_jax.train_jax --mode throughput --out experiments/competition_native_jax/throughput --manifest experiments/manifests/competition_native_jax_throughput_ladder_v2.json

python - <<'PY'
import json
from pathlib import Path
m = json.loads(Path("experiments/manifests/competition_native_jax_throughput_ladder_v2.json").read_text())
cfg = m.get("frozen_config") or {}
Path("experiments/competition_native_jax/frozen_train_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("FROZEN", cfg, flush=True)
if not cfg:
    raise SystemExit("BLOCKED_COMPUTE: no frozen throughput config")
PY

NUM_ENVS=$(python -c "import json; print(json.load(open('experiments/competition_native_jax/frozen_train_config.json'))['num_envs'])")
ROLLOUT=$(python -c "import json; print(json.load(open('experiments/competition_native_jax/frozen_train_config.json'))['rollout_len'])")
SHORT_BUDGET=$(python -c "import json; print(json.load(open('experiments/manifests/competition_native_jax_throughput_ladder_v2.json'))['short_day_budget_transitions'])")
MED_BUDGET=$(python -c "import json; print(json.load(open('experiments/manifests/competition_native_jax_throughput_ladder_v2.json'))['medium_day_budget_transitions'])")

echo "=== R-E.5 smoke envs=${NUM_ENVS} rollout=${ROLLOUT} ==="
python -u -m train.competition_native_jax.train_jax --mode smoke --out experiments/competition_native_jax/smoke_jax --num-envs "${NUM_ENVS}" --rollout-len "${ROLLOUT}"

# Skip short/medium if smoke TPS implies insufficient budget
python - <<PY
import json, math
from pathlib import Path
m = json.loads(Path("experiments/manifests/competition_native_jax_throughput_ladder_v2.json").read_text())
smoke = json.loads(Path("experiments/competition_native_jax/smoke_jax/smoke_report.json").read_text())
tps = float(m.get("valid_learning_tps") or smoke.get("valid_learning_tps") or 0)
short_budget = int(m.get("short_day_budget_transitions") or 0)
print(f"TPS={tps} short_budget={short_budget}", flush=True)
if tps < 0.05 or short_budget < 32:
    Path("experiments/manifests/competition_native_jax_daytime_training_summary.json").write_text(
        json.dumps({
            "schema_version": 1,
            "kind": "COMPETITION_NATIVE_JAX_DAYTIME_TRAINING_SUMMARY",
            "status": "BLOCKED_COMPUTE",
            "reason": f"insufficient_tps={tps}_short_budget={short_budget}",
            "smoke": smoke,
            "throughput": m,
            "short_day": {"status": "SKIPPED_WITH_REASON", "reason": "BLOCKED_COMPUTE"},
            "medium_day": {"status": "SKIPPED_WITH_REASON", "reason": "BLOCKED_COMPUTE"},
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    raise SystemExit(42)
PY

echo "=== R-E.6 short budget=${SHORT_BUDGET} ==="
python -u -m train.competition_native_jax.train_jax --mode short --out experiments/competition_native_jax/short --num-envs "${NUM_ENVS}" --rollout-len "${ROLLOUT}" --budget-transitions "${SHORT_BUDGET}"

echo "=== R-E.7 medium budget=${MED_BUDGET} ==="
python -u -m train.competition_native_jax.train_jax --mode medium --out experiments/competition_native_jax/medium --num-envs "${NUM_ENVS}" --rollout-len "${ROLLOUT}" --budget-transitions "${MED_BUDGET}"

echo "R-E complete"
