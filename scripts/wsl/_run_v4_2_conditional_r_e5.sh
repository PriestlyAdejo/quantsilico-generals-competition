#!/usr/bin/env bash
# Conditional R-E.5 smoke after V4.2 systems promotion
set -euo pipefail
REPO_ROOT="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${REPO_ROOT}/third_party/generals-bots"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="${HOME}/quantsilico-runtime/jax_cache"

GATE="experiments/manifests/competition_native_jax_v4_2_systems_promotion_gate.json"
python - <<'PY'
import json, sys
from pathlib import Path
g = json.loads(Path("experiments/manifests/competition_native_jax_v4_2_systems_promotion_gate.json").read_text())
if not g.get("ready"):
    print("NOT_READY", g.get("blocker"))
    sys.exit(2)
print("READY", g.get("selected"), g.get("valid_learning_tps"))
PY

# Prefer existing resume script if present
if [[ -f scripts/wsl/run_quantsilico_resume_r_e.sh ]]; then
  echo "=== Invoking existing R-E resume helper ==="
  # Pass V4.2 selected config via env
  export QS_V42_PARENT_GATE="${GATE}"
  bash scripts/wsl/run_quantsilico_resume_r_e.sh
else
  echo "=== Inline R-E.5 smoke (100k transitions / 30 min first limit) ==="
  python - <<'PY'
import json
from pathlib import Path
from train.competition_native_jax.train_jax import _train_loop
g = json.loads(Path("experiments/manifests/competition_native_jax_v4_2_systems_promotion_gate.json").read_text())
cfg = g["selected"]
rep = _train_loop(
    Path("experiments/competition_native_jax/v4_2_smoke_r_e5"),
    kind="smoke_r_e5",
    max_transitions=100_000,
    max_updates=10_000,
    max_seconds=30 * 60,
    seed=0,
    **cfg,
)
Path("experiments/manifests/competition_native_jax_v4_2_smoke_r_e5.json").write_text(json.dumps(rep, indent=2)+"\n")
print(json.dumps({"status": rep["status"], "transitions": rep["transitions"], "tps": rep["valid_learning_tps"]}, indent=2))
PY
fi
