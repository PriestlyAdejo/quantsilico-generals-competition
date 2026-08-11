#!/usr/bin/env bash
set -euo pipefail
# Do not pkill patterns that match this script's own argv.
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
exec bash /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition/scripts/wsl/_run_v4_1_continue.sh
