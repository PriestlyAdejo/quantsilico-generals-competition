#!/usr/bin/env bash
set -euo pipefail
pkill -f 'run_competition_native_jax_daytime_eval' 2>/dev/null || true
pkill -f '_run_v4_3_daytime_eval' 2>/dev/null || true
sleep 1
nvidia-smi --query-gpu=memory.used --format=csv,noheader || true
pgrep -af 'daytime_eval|r_e6.py' || echo CLEAR
