#!/usr/bin/env bash
set -euo pipefail
REPO="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
tr -d '\r' < "${REPO}/scripts/wsl/_run_v4_2_continue.sh" > /tmp/_run_v4_2_continue.sh
mkdir -p "${REPO}/experiments/logs/owned_jobs"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="${REPO}/experiments/logs/owned_jobs/v42_continue_${STAMP}.out.log"
ERR="${REPO}/experiments/logs/owned_jobs/v42_continue_${STAMP}.err.log"
nohup bash /tmp/_run_v4_2_continue.sh >"${OUT}" 2>"${ERR}" &
echo $! > "${REPO}/experiments/logs/owned_jobs/v42_continue.pid"
echo "${OUT}" > "${REPO}/experiments/logs/owned_jobs/v42_continue.outpath"
echo "STARTED_PID=$(cat ${REPO}/experiments/logs/owned_jobs/v42_continue.pid)"
echo "OUT=${OUT}"
