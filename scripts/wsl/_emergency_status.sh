#!/usr/bin/env bash
set -euo pipefail
REPO=/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition
cd "$REPO"
echo "=== PROCS ==="
pgrep -af 'emergency_bootstrap_canary|emergency_cpu_ops|emergency_exact_resume|emergency_sprint_watchdog' || echo NONE
echo "=== OPS LOG TAIL ==="
ls -t experiments/logs/owned_jobs/emergency_ops_*.out.log 2>/dev/null | head -1 | xargs -I{} tail -n 40 {}
echo "=== CANARY JSON ==="
if [[ -f experiments/manifests/emergency_bootstrap_canary.json ]]; then
  python3 - <<'PY'
import json
d=json.load(open("experiments/manifests/emergency_bootstrap_canary.json"))
print(d.get("status"), "games", d.get("new_games_played"), "score", d.get("canary_score"), "hunter_wins", d.get("hunter_wins"))
PY
else
  echo NO_CANARY_JSON
fi
echo "=== LATEST ==="
python3 - <<'PY'
import json
from pathlib import Path
p=Path.home()/"quantsilico-runtime/emergency_rolling_v1/training/metrics/emergency_training_latest.json"
if p.exists():
  d=json.loads(p.read_text())
  print("updates", d.get("updates"), "tps", d.get("tps"), "transitions", d.get("transitions"))
else:
  print("NO_LATEST")
PY
echo "=== CKPTS ==="
ls /home/pries/quantsilico-runtime/emergency_rolling_v1/training/checkpoints/ 2>/dev/null || true
echo "=== DASHBOARD GATE ==="
test -f experiments/manifests/emergency_dashboard_gate.json && cat experiments/manifests/emergency_dashboard_gate.json || echo NO_DASH
echo "=== UPLOAD_THIS ==="
head -n 20 submission/EMERGENCY_UPLOAD_THIS.md 2>/dev/null || echo NO_UPLOAD_THIS
