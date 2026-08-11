#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition
python3 scripts/_fix_emergency_sh_lf.py
chmod +x scripts/wsl/_emergency_cpu_ops.sh scripts/wsl/_emergency_sprint_watchdog.sh
mkdir -p experiments/logs/owned_jobs
nohup bash scripts/wsl/_emergency_cpu_ops.sh > experiments/logs/owned_jobs/emergency_ops_launch.out.log 2>&1 &
echo OPS_PID=$!
nohup bash scripts/wsl/_emergency_sprint_watchdog.sh > /dev/null 2>&1 &
echo WD_PID=$!
sleep 3
pgrep -af 'emergency_cpu_ops|emergency_sprint_watchdog|emergency_exact_resume' || true
tail -n 8 experiments/logs/owned_jobs/emergency_ppo_20260807_022105.out.log || true
