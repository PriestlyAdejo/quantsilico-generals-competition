#!/usr/bin/env bash
set -euo pipefail
REPO="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
tr -d '\r' < "${REPO}/scripts/wsl/_run_v4_2_r_e5.sh" > /tmp/_run_v4_2_r_e5.sh
mkdir -p "${REPO}/experiments/logs/owned_jobs"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="${REPO}/experiments/logs/owned_jobs/v42_re5_${STAMP}.out.log"
ERR="${REPO}/experiments/logs/owned_jobs/v42_re5_${STAMP}.err.log"
nohup bash /tmp/_run_v4_2_r_e5.sh >"${OUT}" 2>"${ERR}" &
echo $! > "${REPO}/experiments/logs/owned_jobs/v42_re5.pid"
echo "STARTED_PID=$(cat ${REPO}/experiments/logs/owned_jobs/v42_re5.pid)"
echo "OUT=${OUT}"
