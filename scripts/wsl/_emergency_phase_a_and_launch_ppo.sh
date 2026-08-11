#!/usr/bin/env bash
# Emergency Phase A bootstrap (CPU JAX) then launch exact-resume PPO on GPU.
set -euo pipefail
REPO_ROOT="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${REPO_ROOT}/third_party/generals-bots"
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="${HOME}/quantsilico-runtime/jax_cache"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" experiments/logs/owned_jobs experiments/manifests

STAMP="$(date -u +%Y%m%d_%H%M%S)"
BOOT_LOG="experiments/logs/owned_jobs/emergency_phase_a_${STAMP}.out.log"

echo "=== Emergency Phase A ${STAMP} ===" | tee "${BOOT_LOG}"
# Bootstrap must not steal GPU
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu python scripts/emergency_rolling_bootstrap_phase_a.py 2>&1 | tee -a "${BOOT_LOG}"
RC=${PIPESTATUS[0]}
if [[ "${RC}" -ne 0 ]]; then
  echo "PHASE_A_FAILED rc=${RC}" | tee -a "${BOOT_LOG}"
  exit "${RC}"
fi

# Optional training worktree (best-effort; ops stays on this checkout)
WT_PARENT="$(dirname "${REPO_ROOT}")"
WT="${WT_PARENT}/quantsilico-emergency-training"
if [[ ! -d "${WT}" ]]; then
  git worktree add "${WT}" HEAD 2>&1 | tee -a "${BOOT_LOG}" || echo "WORKTREE_ADD_SKIPPED" | tee -a "${BOOT_LOG}"
fi
if [[ -d "${WT}" ]]; then
  python - <<PY 2>&1 | tee -a "${BOOT_LOG}"
import json
from pathlib import Path
from datetime import datetime, timezone
p = Path("experiments/manifests/emergency_rolling_programme_state.json")
doc = json.loads(p.read_text())
doc["worktrees"]["emergency_training"] = "/mnt/c/Users/pries/Documents/Projects/quantsilico-emergency-training"
# Prefer Windows-visible path if worktree is under /mnt/c
import os
wt = Path("/mnt/c/Users/pries/Documents/Projects/quantsilico-emergency-training")
if wt.exists():
    doc["worktrees"]["emergency_training"] = str(wt)
doc["updated_at"] = datetime.now(timezone.utc).isoformat()
tmp = p.with_suffix(".json.tmp")
tmp.write_text(json.dumps(doc, indent=2) + "\n")
tmp.replace(p)
print("WORKTREE_RECORDED", doc["worktrees"])
PY
fi

echo "=== Launching exact-resume PPO ===" | tee -a "${BOOT_LOG}"
TRAIN_LOG="experiments/logs/owned_jobs/emergency_ppo_${STAMP}.out.log"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
# Train on GPU — do NOT set JAX_PLATFORMS=cpu
unset JAX_PLATFORMS || true
unset CUDA_VISIBLE_DEVICES || true
nohup python scripts/emergency_exact_resume_ppo.py > "${TRAIN_LOG}" 2>&1 &
TPID=$!
echo "${TPID}" > experiments/logs/owned_jobs/emergency_ppo.pid
python - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone
pid = int(Path("experiments/logs/owned_jobs/emergency_ppo.pid").read_text().strip())
owned = {
  "schema_version": 1,
  "kind": "COMPETITION_NATIVE_JAX_OWNED_PROCESSES",
  "updated_at": datetime.now(timezone.utc).isoformat(),
  "jobs": [{
    "job_id": "emergency_ppo",
    "stage": "EMERGENCY_EXACT_RESUME",
    "wsl_linux_pid": pid,
    "status": "RUNNING",
    "stdout_path": "${TRAIN_LOG}",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "termination_command": f"kill -INT {pid}",
  }],
}
Path("experiments/manifests/competition_native_jax_owned_processes.json").write_text(json.dumps(owned, indent=2)+"\n")
print("STARTED_EMERGENCY_PPO", pid)
PY
sleep 25
tail -n 40 "${TRAIN_LOG}" || true
echo "BOOT_DONE train_log=${TRAIN_LOG} pid=$(cat experiments/logs/owned_jobs/emergency_ppo.pid)"
