#!/usr/bin/env bash
# Gracefully stop superseded V4.1 continue worker
set -uo pipefail
REPO="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
SCALE="${REPO}/experiments/competition_native_jax/v4_1_scaling"
RUNTIME="${HOME}/quantsilico-runtime/competition_native_jax"

echo "classification=RUNNING_SCALING"
echo "exit_reason=SUPERSEDED_BY_V4_2_OPERATOR_PROGRAMME"

# 1) STOP_REQUEST in all scaling out dirs + runtime
for d in "${SCALE}"/*/ "${RUNTIME}"/probe_v41_*; do
  if [[ -d "$d" ]]; then
    touch "${d}/STOP_REQUEST"
    echo "wrote STOP_REQUEST -> $d"
  fi
done

BASH_PID=385
PY_PID=916

alive() { kill -0 "$1" 2>/dev/null; }

wait_dead() {
  local pid=$1 max=$2
  local i=0
  while (( i < max )); do
    if ! alive "$pid"; then
      echo "pid $pid dead after ${i}s"
      return 0
    fi
    sleep 5
    i=$((i+5))
    echo "waiting pid=$pid t=${i}s"
  done
  return 1
}

echo "=== phase STOP_REQUEST wait 180s ==="
if wait_dead "$PY_PID" 180; then
  echo "stopped via STOP_REQUEST"
else
  echo "=== SIGINT python $PY_PID and bash $BASH_PID ==="
  kill -INT "$PY_PID" 2>/dev/null || true
  kill -INT "$BASH_PID" 2>/dev/null || true
  if ! wait_dead "$PY_PID" 180; then
    echo "=== SIGTERM ==="
    kill -TERM "$PY_PID" 2>/dev/null || true
    kill -TERM "$BASH_PID" 2>/dev/null || true
    if ! wait_dead "$PY_PID" 60; then
      echo "=== SIGKILL ==="
      kill -KILL "$PY_PID" 2>/dev/null || true
      kill -KILL "$BASH_PID" 2>/dev/null || true
      sleep 2
    fi
  fi
fi

# Also reap bash if python gone but bash remains
if alive "$BASH_PID"; then
  echo "bash still alive; SIGTERM then SIGKILL"
  kill -TERM "$BASH_PID" 2>/dev/null || true
  sleep 5
  kill -KILL "$BASH_PID" 2>/dev/null || true
fi

echo "=== final check ==="
pgrep -af '_run_v4_1|probe_v41' || echo "NO_V41_WORKER"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true
echo "DONE_STOP"
