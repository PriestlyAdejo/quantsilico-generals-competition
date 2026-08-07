#!/usr/bin/env bash
set -euo pipefail

readonly REPO=/workspace/quantsilico-generals
readonly RUNTIME=/workspace/quantsilico-runtime/cloud_gpu_last_push_v1
readonly PYTHON=/tmp/qs-venv312/bin/python
readonly PARENT="$RUNTIME/geometry_gate/e512_r32/ckpt_final"
readonly STOP_AT=2026-08-08T00:32:00Z
readonly LOG="$RUNTIME/training/logs/cloud_train.log"

if tmux has-session -t cloud_train 2>/dev/null; then
  echo "cloud_train already exists" >&2
  exit 2
fi
if pgrep -af 'cloud_gpu_last_push.py.* train' >/dev/null; then
  echo "a cloud trainer is already running" >&2
  exit 3
fi
test -f "$PARENT/COMPLETE"
test -f "$PARENT/sha256_manifest.json"
test ! -e "$RUNTIME/training/STOP_REQUEST"
mkdir -p "$RUNTIME/training/logs" "$RUNTIME/jax_cache"

tmux new-session -d -s cloud_train \
  "cd '$REPO' && exec env \
PYTHONPATH=src:. \
JAX_COMPILATION_CACHE_DIR='$RUNTIME/jax_cache' \
JAX_ENABLE_COMPILATION_CACHE=true \
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
'$PYTHON' -u scripts/cloud_gpu_last_push.py \
--runtime '$RUNTIME' train \
--parent '$PARENT' \
--num-envs 512 --rollout-len 32 \
--stop-at '$STOP_AT' >> '$LOG' 2>&1"

sleep 2
tmux has-session -t cloud_train
trainer_pid="$(tmux display-message -p -t cloud_train '#{pane_pid}')"
trainer_command="$(tr '\0' ' ' < "/proc/$trainer_pid/cmdline")"
case "$trainer_command" in
  *cloud_gpu_last_push.py*" train "*) ;;
  *)
    echo "unexpected trainer command: $trainer_command" >&2
    exit 4
    ;;
esac
printf 'CLOUD_TRAIN_LAUNCHED pid=%s stop_at=%s\n' "$trainer_pid" "$STOP_AT"
