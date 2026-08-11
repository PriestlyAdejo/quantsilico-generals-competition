#!/usr/bin/env bash
# R-E.6 short daytime PPO — cold restart, complete-update budget, full checkpoints.
set -euo pipefail
REPO_ROOT="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${REPO_ROOT}/third_party/generals-bots"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="${HOME}/quantsilico-runtime/jax_cache"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" experiments/logs/owned_jobs experiments/manifests

STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG="experiments/logs/owned_jobs/v43_re6_${STAMP}.out.log"
exec > >(tee -a "${LOG}") 2>&1

echo "=== R-E.6 short daytime ${STAMP} ==="
python scripts/_run_v4_3_r_e6.py
RC=$?

python - <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone
Path("experiments/manifests/competition_native_jax_owned_processes.json").write_text(json.dumps({
  "schema_version": 1, "kind": "COMPETITION_NATIVE_JAX_OWNED_PROCESSES",
  "updated_at": datetime.now(timezone.utc).isoformat(), "jobs": []
}, indent=2)+"\n")
PY
exit "${RC}"
