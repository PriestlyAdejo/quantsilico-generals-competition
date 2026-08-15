#!/usr/bin/env bash
# Remote GPU validation bootstrap for the QuantSilico Marathon (run ON the pod).
# Verifies: JAX CUDA backend with no silent fallback, device count, checkpoint
# load, genuine resume semantics (ratio 1.0) via a bounded 3-update smoke.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/repo}"
CKPT_DIR="${CKPT_DIR:-/workspace/ckpt_baseline_v0}"
VENV_BIN="${VENV_BIN:-${REPO_DIR}/.venv312/bin}"
ARM_ID="${ARM_ID:-gpu-smoke}"
MAX_UPDATES="${MAX_UPDATES:-3}"
NUM_ENVS="${NUM_ENVS:-8}"
ROLLOUT_LEN="${ROLLOUT_LEN:-16}"

cd "${REPO_DIR}"

echo "=== device check ==="
"${VENV_BIN}/python" - <<'PY'
import jax

print("JAX", jax.__version__, "backend", jax.default_backend(), "devices", jax.devices())
assert jax.default_backend() == "gpu", "SILENT_CPU_FALLBACK_DETECTED"
assert len(jax.devices()) == 1, f"unexpected device count {len(jax.devices())}"
kind = jax.devices()[0].device_kind
print("device_kind", kind)
PY

echo "=== nvidia-smi snapshot ==="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo "=== checkpoint present ==="
ls -l "${CKPT_DIR}"

echo "=== resume smoke (${ARM_ID}, ${MAX_UPDATES} updates) ==="
PYTHONPATH="${REPO_DIR}:${REPO_DIR}/src:${REPO_DIR}/third_party/generals-bots" "${VENV_BIN}/python" scripts/training/run_sh_r1_arm.py \
  --arm-id "${ARM_ID}" \
  --num-envs "${NUM_ENVS}" \
  --rollout-len "${ROLLOUT_LEN}" \
  --checkpoint "${CKPT_DIR}" \
  --max-updates "${MAX_UPDATES}" \
  --out-dir /workspace/screening_runs/${ARM_ID}

echo "=== post-smoke nvidia-smi ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
