#!/usr/bin/env bash
set -euo pipefail
pgrep -af 'run_competition_native_jax_daytime_eval|GeneralsEnv' || echo NO_EVAL
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
wc -l /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition/experiments/logs/owned_jobs/v43_eval_cpu.out.log
ls -la /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition/experiments/manifests/competition_native_jax_daytime_eval_screening.partial.json 2>/dev/null || echo NO_PARTIAL
