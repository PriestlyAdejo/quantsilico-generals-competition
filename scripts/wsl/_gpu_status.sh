#!/usr/bin/env bash
set -euo pipefail
# Kill stray python/jax holders on this GPU before V4.1 (owned job only)
nvidia-smi || true
# List python PIDs using GPU
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
