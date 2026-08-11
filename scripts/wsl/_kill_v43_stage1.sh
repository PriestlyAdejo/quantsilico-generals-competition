#!/usr/bin/env bash
set -euo pipefail
pkill -f '_stage1_v4_3_stage3b.py' 2>/dev/null || true
pkill -f '_run_v4_3_stage3a_3b.sh' 2>/dev/null || true
sleep 2
pgrep -af '_stage1_v4_3|_run_v4_3_stage3a' || echo KILLED
nvidia-smi --query-gpu=memory.used --format=csv,noheader
