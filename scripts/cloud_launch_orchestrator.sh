#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 TRAINER_PID" >&2
  exit 2
fi

readonly TRAINER_PID="$1"
readonly REPO=/workspace/quantsilico-generals
readonly RUNTIME=/workspace/quantsilico-runtime/cloud_gpu_last_push_v1
readonly PYTHON=/tmp/qs-venv312/bin/python
readonly STOP_AT=2026-08-08T00:32:00Z
readonly LOG="$RUNTIME/training/logs/cloud_orchestrator.log"

test -r "/proc/$TRAINER_PID/cmdline"
trainer_command="$(tr '\0' ' ' < "/proc/$TRAINER_PID/cmdline")"
case "$trainer_command" in
  *cloud_gpu_last_push.py*" train "*) ;;
  *)
    echo "PID is not the authoritative trainer: $trainer_command" >&2
    exit 3
    ;;
esac
if tmux has-session -t cloud_orchestrator 2>/dev/null; then
  echo "cloud_orchestrator already exists" >&2
  exit 4
fi

tmux new-session -d -s cloud_orchestrator \
  "cd '$REPO' && exec '$PYTHON' -u scripts/cloud_orchestrator.py \
--runtime '$RUNTIME' --trainer-pid '$TRAINER_PID' \
--stop-at '$STOP_AT' --poll-seconds 30 >> '$LOG' 2>&1"

sleep 2
tmux has-session -t cloud_orchestrator
printf 'CLOUD_ORCHESTRATOR_LAUNCHED trainer_pid=%s stop_at=%s\n' "$TRAINER_PID" "$STOP_AT"
