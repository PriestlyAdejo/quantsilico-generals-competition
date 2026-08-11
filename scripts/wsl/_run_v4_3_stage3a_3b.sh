#!/usr/bin/env bash
# V4.3 Stage 1: exact-hash Stage 3A then Stage 3B on selected daytime snapshot.
# Does NOT overwrite frozen V4 / V4.1 / v4_2 ladder or end_to_end_audit artefacts.
set -euo pipefail
REPO_ROOT="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${REPO_ROOT}/third_party/generals-bots"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="${HOME}/quantsilico-runtime/jax_cache"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" experiments/manifests experiments/reports experiments/logs/owned_jobs

STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG="experiments/logs/owned_jobs/v43_stage3a3b_${STAMP}.out.log"
exec > >(tee -a "${LOG}") 2>&1

echo "=== V4.3 Stage 1 exact-hash 3A+3B starting ${STAMP} ==="

python - <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone
from train.competition_native_jax.train_jax import lineage_hashes, detect_jax_device

snap = json.loads(Path("experiments/manifests/competition_native_jax_v4_3_source_snapshot.json").read_text())
lin = lineage_hashes()
assert lin["env_implementation_hash"] == snap["env_implementation_hash"], (
    lin["env_implementation_hash"], snap["env_implementation_hash"]
)
assert lin["learner_implementation_hash"] == snap["learner_implementation_hash"]
assert lin["env_semantics_hash"] == snap["env_semantics_hash"]
print(json.dumps({"lineage_ok": True, "lineage": lin, "device": detect_jax_device()}, indent=2), flush=True)

# Register owned job
owned_path = Path("experiments/manifests/competition_native_jax_owned_processes.json")
owned = json.loads(owned_path.read_text()) if owned_path.exists() else {"schema_version": 1, "kind": "COMPETITION_NATIVE_JAX_OWNED_PROCESSES", "jobs": []}
owned["updated_at"] = datetime.now(timezone.utc).isoformat()
owned["jobs"] = [j for j in owned.get("jobs", []) if j.get("id") != "v43_stage3a3b"]
owned["jobs"].append({
    "id": "v43_stage3a3b",
    "kind": "STAGE_3A_3B_EXACT_HASH",
    "status": "RUNNING",
    "started_at": owned["updated_at"],
})
owned_path.write_text(json.dumps(owned, indent=2) + "\n")
PY

echo "=== Stage 3A pytest ==="
pytest -q tests/competition_native_jax -o cache_dir=/tmp/qs_pytest_cache --tb=short \
  2>&1 | tee experiments/manifests/_v4_3_stage3a_pytest.log
PYTEST_RC=${PIPESTATUS[0]}
if [[ "${PYTEST_RC}" -ne 0 ]]; then
  python - <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone
from train.competition_native_jax.train_jax import lineage_hashes
rep = {
  "schema_version": 1,
  "kind": "COMPETITION_NATIVE_JAX_V4_2_STAGE3A_EXACT_HASH",
  "status": "FAILED",
  "pytest_rc": 1,
  "log": "experiments/manifests/_v4_3_stage3a_pytest.log",
  "lineage": lineage_hashes(),
  "updated_at": datetime.now(timezone.utc).isoformat(),
}
Path("experiments/manifests/competition_native_jax_v4_2_stage3a_exact_hash.json").write_text(json.dumps(rep, indent=2)+"\n")
prog = json.loads(Path("experiments/manifests/competition_native_jax_v4_3_programme_state.json").read_text())
prog["status"] = "BLOCKED_V4_2_STAGE3A"
prog["current_stage"] = "STAGE_1_FAILED_3A"
Path("experiments/manifests/competition_native_jax_v4_3_programme_state.json").write_text(json.dumps(prog, indent=2)+"\n")
raise SystemExit(1)
PY
fi

python - <<'PY'
import json, re
from pathlib import Path
from datetime import datetime, timezone
from train.competition_native_jax.train_jax import lineage_hashes

log = Path("experiments/manifests/_v4_3_stage3a_pytest.log").read_text(encoding="utf-8", errors="replace")
m = re.search(r"(\d+) passed", log)
n_pass = int(m.group(1)) if m else None
rep = {
  "schema_version": 1,
  "kind": "COMPETITION_NATIVE_JAX_V4_2_STAGE3A_EXACT_HASH",
  "status": "PASSED",
  "gate": "STAGE_3A_EXACT_HASH_PASS",
  "pytest": f"{n_pass} passed" if n_pass is not None else "passed",
  "pytest_passed_count": n_pass,
  "log": "experiments/manifests/_v4_3_stage3a_pytest.log",
  "selected_env_implementation_hash": lineage_hashes()["env_implementation_hash"],
  "lineage": lineage_hashes(),
  "updated_at": datetime.now(timezone.utc).isoformat(),
}
Path("experiments/manifests/competition_native_jax_v4_2_stage3a_exact_hash.json").write_text(json.dumps(rep, indent=2)+"\n")
# Also refresh legacy parity_3a pointer without claiming old unstamped artefact is authoritative
Path("experiments/manifests/competition_native_jax_parity_3a.json").write_text(json.dumps({
  "schema_version": 1,
  "kind": "COMPETITION_NATIVE_JAX_PARITY_3A",
  "status": "PASSED",
  "pytest": rep["pytest"],
  "env_implementation_hash": rep["selected_env_implementation_hash"],
  "source": "competition_native_jax_v4_2_stage3a_exact_hash.json",
}, indent=2)+"\n")
print(json.dumps(rep, indent=2), flush=True)
PY

echo "=== Stage 3B full exact-hash parity ==="
python scripts/_stage1_v4_3_stage3b.py
STAGE3B_RC=$?

python - <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone
owned_path = Path("experiments/manifests/competition_native_jax_owned_processes.json")
owned = json.loads(owned_path.read_text())
owned["updated_at"] = datetime.now(timezone.utc).isoformat()
owned["jobs"] = []
owned_path.write_text(json.dumps(owned, indent=2) + "\n")
PY

if [[ "${STAGE3B_RC}" -ne 0 ]]; then
  echo "STAGE_3B_FAILED"
  exit 1
fi
echo "STAGE_1_3A_3B_DONE"
