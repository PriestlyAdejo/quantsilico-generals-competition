#!/usr/bin/env bash
# R-E entrypoint after GPU_JAX_VERIFIED (locked order).
set -euo pipefail
REPO_ROOT="${1:-}"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

echo "=== R-E.1 impacted regressions ==="
python -m pytest tests/competition_native_jax -q --tb=line

echo "=== R-E.2 GPU zero-update correctness ==="
python -m train.competition_native_jax.train_jax --mode correctness --out experiments/competition_native_jax/gpu_correctness

echo "=== R-E.3 tiny ==="
python -m train.competition_native_jax.train_jax --mode tiny --out experiments/competition_native_jax/tiny

echo "=== R-E.4 throughput ladder freeze ==="
python -m train.competition_native_jax.train_jax --mode throughput --out experiments/competition_native_jax/throughput \
  --manifest experiments/manifests/competition_native_jax_throughput_ladder_v2.json

python - <<'PY'
import json
from pathlib import Path
m = json.loads(Path("experiments/manifests/competition_native_jax_throughput_ladder_v2.json").read_text())
cfg = m.get("frozen_config") or {}
Path("experiments/competition_native_jax/frozen_train_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("FROZEN", cfg)
if not cfg:
    raise SystemExit("BLOCKED_COMPUTE: no frozen throughput config")
PY

NUM_ENVS=$(python -c "import json; print(json.load(open('experiments/competition_native_jax/frozen_train_config.json'))['num_envs'])")
ROLLOUT=$(python -c "import json; print(json.load(open('experiments/competition_native_jax/frozen_train_config.json'))['rollout_len'])")
SHORT_BUDGET=$(python -c "import json; print(json.load(open('experiments/manifests/competition_native_jax_throughput_ladder_v2.json'))['short_day_budget_transitions'])")
MED_BUDGET=$(python -c "import json; print(json.load(open('experiments/manifests/competition_native_jax_throughput_ladder_v2.json'))['medium_day_budget_transitions'])")

echo "=== R-E.5 smoke with frozen config envs=${NUM_ENVS} rollout=${ROLLOUT} ==="
python -m train.competition_native_jax.train_jax --mode smoke --out experiments/competition_native_jax/smoke_jax \
  --num-envs "${NUM_ENVS}" --rollout-len "${ROLLOUT}"

echo "=== R-E.6 short budget=${SHORT_BUDGET} ==="
python -m train.competition_native_jax.train_jax --mode short --out experiments/competition_native_jax/short \
  --num-envs "${NUM_ENVS}" --rollout-len "${ROLLOUT}" --budget-transitions "${SHORT_BUDGET}"

echo "=== R-E.7 medium budget=${MED_BUDGET} ==="
python -m train.competition_native_jax.train_jax --mode medium --out experiments/competition_native_jax/medium \
  --num-envs "${NUM_ENVS}" --rollout-len "${ROLLOUT}" --budget-transitions "${MED_BUDGET}"

echo "R-E complete"
