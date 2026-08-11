#!/usr/bin/env bash
set -euo pipefail
echo "=== all relevant ==="
ps -eo pid,ppid,etime,stat,cmd | grep -E 'v4_1|train_jax|probe_v41|_train_loop|competition_native_jax|python' | grep -v grep | head -40
echo "=== nvidia ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || nvidia-smi | head -40
echo "=== newest scaling heartbeats ==="
for d in /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition/experiments/competition_native_jax/v4_1_scaling/*/; do
  echo "-- $d"
  ls -la "$d" 2>/dev/null | head -15
  if [[ -f "${d}heartbeat.json" ]]; then cat "${d}heartbeat.json"; echo; fi
done
echo "=== STOP places ==="
ls /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition/experiments/competition_native_jax/v4_1_scaling/*/ 2>/dev/null
