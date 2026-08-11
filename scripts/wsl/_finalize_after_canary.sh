#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition
# Poll up to ~35 minutes
for i in $(seq 1 70); do
  if [[ -f experiments/manifests/emergency_bootstrap_canary.json ]]; then
    echo CANARY_READY
    break
  fi
  if ! pgrep -f 'python scripts/emergency_bootstrap_canary.py' >/dev/null 2>&1; then
    echo CANARY_PROC_GONE
    sleep 2
    break
  fi
  echo "waiting_canary_$i"
  sleep 30
done
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
export PYTHONPATH="${PWD}/src:${PWD}"
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
python scripts/emergency_write_dashboard_gate.py
python scripts/emergency_final_selection_gate.py
python scripts/emergency_patch_programme_state.py
bash scripts/wsl/_emergency_status.sh
echo FINISH_WAIT_DONE
