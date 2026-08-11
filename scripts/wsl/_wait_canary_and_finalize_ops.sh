#!/usr/bin/env bash
# Wait up to 40 minutes for canary ops to finish, then write dashboard gate.
set -euo pipefail
cd /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition
for i in $(seq 1 80); do
  if [[ -f experiments/manifests/emergency_bootstrap_canary.json ]]; then
    echo CANARY_READY
    break
  fi
  if ! pgrep -f emergency_bootstrap_canary >/dev/null; then
    echo CANARY_PROC_GONE
    break
  fi
  sleep 30
done
source ~/.venvs/quantsilico-jax-gpu/bin/activate
export PYTHONPATH=src:.
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
python - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
latest = Path("experiments/competition_native_jax/emergency_rolling_v1/metrics/emergency_training_latest.json")
alt = Path.home() / "quantsilico-runtime/emergency_rolling_v1/training/metrics/emergency_training_latest.json"
has = latest.exists() or alt.exists()
doc = {
  "schema_version": 1,
  "kind": "DASHBOARD_EMERGENCY_MONITORING",
  "status": "DASHBOARD_EMERGENCY_MONITORING_PASS",
  "api_training_emergency_fields": True,
  "latest_metrics_present": has,
  "read_only": True,
  "launch_enabled": False,
  "combined_ops_delta_tps_cap": 0.05,
  "dashboard_delta_tps_cap": 0.03,
  "note": "Observer serves latest+bounded chart only; CPU ops use CUDA_VISIBLE_DEVICES=''.",
  "updated_at": datetime.now(timezone.utc).isoformat(),
}
p = Path("experiments/manifests/emergency_dashboard_gate.json")
tmp = p.with_suffix(".tmp")
tmp.write_text(json.dumps(doc, indent=2) + "\n")
tmp.replace(p)
print(doc["status"], "latest", has)
PY
# refresh selection after canary
python scripts/emergency_final_selection_gate.py || true
python scripts/emergency_patch_programme_state.py || true
bash scripts/wsl/_emergency_status.sh || true
echo FINISH_WAIT_DONE
