#!/usr/bin/env bash
# Resume Stage 3B only (Stage 3A already PASSED on selected hash).
set -euo pipefail
REPO_ROOT="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${REPO_ROOT}/third_party/generals-bots"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="${HOME}/quantsilico-runtime/jax_cache"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" experiments/logs/owned_jobs

STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG="experiments/logs/owned_jobs/v43_stage3b_${STAMP}.out.log"
exec > >(tee -a "${LOG}") 2>&1

echo "=== V4.3 Stage 3B-only ${STAMP} ==="
python - <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone
from train.competition_native_jax.train_jax import lineage_hashes
a3 = json.loads(Path("experiments/manifests/competition_native_jax_v4_2_stage3a_exact_hash.json").read_text())
assert a3["status"] == "PASSED"
assert a3["selected_env_implementation_hash"] == lineage_hashes()["env_implementation_hash"]
owned_path = Path("experiments/manifests/competition_native_jax_owned_processes.json")
owned = json.loads(owned_path.read_text()) if owned_path.exists() else {"schema_version": 1, "kind": "COMPETITION_NATIVE_JAX_OWNED_PROCESSES", "jobs": []}
owned["updated_at"] = datetime.now(timezone.utc).isoformat()
owned["jobs"] = [{"id": "v43_stage3b", "kind": "STAGE_3B_EXACT_HASH", "status": "RUNNING", "started_at": owned["updated_at"]}]
owned_path.write_text(json.dumps(owned, indent=2) + "\n")
print("3A gate preserved; launching 3B", flush=True)
PY

python scripts/_stage1_v4_3_stage3b.py
RC=$?

python - <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone
owned_path = Path("experiments/manifests/competition_native_jax_owned_processes.json")
owned = {"schema_version": 1, "kind": "COMPETITION_NATIVE_JAX_OWNED_PROCESSES", "updated_at": datetime.now(timezone.utc).isoformat(), "jobs": []}
owned_path.write_text(json.dumps(owned, indent=2) + "\n")
PY

exit "${RC}"
