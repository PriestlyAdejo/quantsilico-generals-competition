#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${REPO_ROOT}/third_party/generals-bots"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="${HOME}/quantsilico-runtime/jax_cache"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG="experiments/logs/owned_jobs/v43_ckpt_rt_${STAMP}.out.log"
mkdir -p experiments/logs/owned_jobs
exec > >(tee -a "${LOG}") 2>&1

echo "=== Stage 5 checkpoint round-trip ${STAMP} ==="
python scripts/_stage5_checkpoint_roundtrip.py
RC=$?
python - <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone
prog = json.loads(Path("experiments/manifests/competition_native_jax_v4_3_programme_state.json").read_text())
rt = json.loads(Path("experiments/manifests/competition_native_jax_v4_3_checkpoint_roundtrip.json").read_text())
prog["checkpoint_roundtrip"] = "experiments/manifests/competition_native_jax_v4_3_checkpoint_roundtrip.json"
prog["checkpoint_roundtrip_status"] = rt["status"]
if rt["status"] == "CHECKPOINT_EXACT_CONTINUATION_PASS":
    prog["r_e7_exact_resume_enabled"] = True
    prog["overnight_parent_eligible"] = True
    prog["status"] = "STAGE_5_COMPLETE"
    prog["current_stage"] = "STAGE_6_R_E6"
else:
    prog["r_e7_exact_resume_enabled"] = False
    prog["overnight_parent_eligible"] = False
    prog["status"] = "STAGE_5_CKPT_FAIL_COLD_ONLY"
    prog["current_stage"] = "STAGE_6_R_E6_COLD_ONLY"
prog["updated_at"] = datetime.now(timezone.utc).isoformat()
Path("experiments/manifests/competition_native_jax_v4_3_programme_state.json").write_text(json.dumps(prog, indent=2)+"\n")
print(json.dumps({"ckpt": rt["status"], "stage": prog["current_stage"]}, indent=2))
PY
exit "${RC}"
