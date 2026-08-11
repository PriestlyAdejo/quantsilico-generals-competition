#!/usr/bin/env bash
set -euo pipefail
pgrep -af '_stage1_v4_3|_run_v4_3_stage3' || echo NO_STAGE3
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
