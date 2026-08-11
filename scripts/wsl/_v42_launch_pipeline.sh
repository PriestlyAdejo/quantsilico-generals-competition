#!/usr/bin/env bash
set -euo pipefail
REPO="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
sed -i 's/\r$//' "${REPO}/scripts/wsl/_run_v4_2_pipeline.sh"
mkdir -p "${REPO}/experiments/logs/owned_jobs"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="${REPO}/experiments/logs/owned_jobs/v42_pipeline_${STAMP}.out.log"
ERR="${REPO}/experiments/logs/owned_jobs/v42_pipeline_${STAMP}.err.log"
HB="${REPO}/experiments/logs/owned_jobs/v42_pipeline_${STAMP}.heartbeat"
nohup bash "${REPO}/scripts/wsl/_run_v4_2_pipeline.sh" >"${OUT}" 2>"${ERR}" &
PID=$!
echo "${PID}" > "${REPO}/experiments/logs/owned_jobs/v42_pipeline.pid"
echo "${OUT}" > "${REPO}/experiments/logs/owned_jobs/v42_pipeline.outpath"
date -u +%Y-%m-%dT%H:%M:%SZ > "${HB}"
echo "STARTED_PID=${PID}"
echo "OUT=${OUT}"
echo "ERR=${ERR}"
