#!/usr/bin/env bash
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
LOG="experiments/logs/owned_jobs/v43a_autotune_${STAMP}.out.log"
exec > >(tee -a "${LOG}") 2>&1

echo "=== V4.3A autotune ${STAMP} ==="
python - <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone
owned = {"schema_version": 1, "kind": "COMPETITION_NATIVE_JAX_OWNED_PROCESSES", "updated_at": datetime.now(timezone.utc).isoformat(),
         "jobs": [{"id": "v43a_autotune", "kind": "V4_3A_AUTOTUNE", "status": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat()}]}
Path("experiments/manifests/competition_native_jax_owned_processes.json").write_text(json.dumps(owned, indent=2)+"\n")
PY

python -m train.competition_native_jax.autotune_v4_3a
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
