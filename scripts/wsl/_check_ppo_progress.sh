#!/usr/bin/env bash
set -euo pipefail
sleep 120
echo "=== LOG ==="
tail -n 12 /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition/experiments/logs/owned_jobs/emergency_ppo_20260807_022105.out.log
echo "=== CKPT ==="
ls -la /home/pries/quantsilico-runtime/emergency_rolling_v1/training/checkpoints/
echo "=== LATEST UPDATES ==="
python3 <<'PY'
import json
from pathlib import Path
p = Path.home() / "quantsilico-runtime/emergency_rolling_v1/training/metrics/emergency_training_latest.json"
print(json.loads(p.read_text())["updates"] if p.exists() else None)
PY
ps -o pid,stat,etime,cmd -p 554 || echo DEAD
