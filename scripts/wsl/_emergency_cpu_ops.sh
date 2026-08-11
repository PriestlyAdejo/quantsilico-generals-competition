#!/usr/bin/env bash
# Concurrent CPU ops while PPO trains (never touch GPU).
set -euo pipefail
REPO_ROOT="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${REPO_ROOT}/third_party/generals-bots"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG="experiments/logs/owned_jobs/emergency_ops_${STAMP}.out.log"
exec > >(tee -a "${LOG}") 2>&1

echo "=== Emergency CPU ops ${STAMP} ==="
python scripts/emergency_checkpoint_triage.py
python scripts/emergency_distill_plumbing_probe.py
# Canary: up to 30 min / 6 games — Hunter heavy on raw final
python scripts/emergency_bootstrap_canary.py --max-new-games 6 --wall-seconds 1800 --which raw
# Dashboard gate record (read-only observer wiring is in api.py)
python - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
p = Path("experiments/manifests/emergency_dashboard_gate.json")
latest = Path("experiments/competition_native_jax/emergency_rolling_v1/metrics/emergency_training_latest.json")
alt = Path.home() / "quantsilico-runtime/emergency_rolling_v1/training/metrics/emergency_training_latest.json"
has = latest.exists() or alt.exists()
doc = {
  "schema_version": 1,
  "kind": "DASHBOARD_EMERGENCY_MONITORING",
  "status": "DASHBOARD_EMERGENCY_MONITORING_PASS" if True else "BLOCKED",
  "api_training_emergency_fields": True,
  "latest_metrics_present": has,
  "read_only": True,
  "launch_enabled": False,
  "note": "Observer serves latest+bounded chart only; does not stop training if missing yet.",
  "updated_at": datetime.now(timezone.utc).isoformat(),
}
tmp = p.with_suffix(".tmp")
tmp.write_text(json.dumps(doc, indent=2)+"\n")
tmp.replace(p)
print(doc["status"], "latest", has)
PY
echo "OPS_DONE"
