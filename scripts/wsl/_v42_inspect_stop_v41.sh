#!/usr/bin/env bash
set -euo pipefail
echo "=== CMD 916 ==="
if [[ -r /proc/916/cmdline ]]; then
  tr '\0' ' ' < /proc/916/cmdline
  echo
  echo "CWD: $(readlink /proc/916/cwd || true)"
fi
echo "=== CMD 385 ==="
if [[ -r /proc/385/cmdline ]]; then
  tr '\0' ' ' < /proc/385/cmdline
  echo
fi
echo "=== runtime dirs ==="
ls -la /home/pries/quantsilico-runtime/competition_native_jax/ 2>/dev/null || true
echo "=== v4_1_scaling ==="
ls -la /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition/experiments/competition_native_jax/v4_1_scaling/ 2>/dev/null || true
echo "=== heartbeats ==="
find /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition/experiments/competition_native_jax \
  /home/pries/quantsilico-runtime/competition_native_jax \
  -name 'heartbeat.json' 2>/dev/null | head -20 || true
echo "=== continue out log tail ==="
tail -40 /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition/experiments/logs/owned_jobs/v41_continue_20260806_154254.out.log 2>/dev/null || true
echo "=== continue err log ==="
tail -40 /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition/experiments/logs/owned_jobs/v41_continue_20260806_154254.err.log 2>/dev/null || true
echo "=== live pids ==="
pgrep -af '_run_v4_1|train_jax|probe_v41' || echo NONE
