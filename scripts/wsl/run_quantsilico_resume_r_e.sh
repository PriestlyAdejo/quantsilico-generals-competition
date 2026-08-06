#!/usr/bin/env bash
# R-E entrypoint after GPU_JAX_VERIFIED.
set -euo pipefail
REPO_ROOT="${1:-}"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
python -m pytest tests/competition_native_jax -q
python - <<'PY'
from pathlib import Path
from train.competition_native_jax.train_jax import run_gpu_correctness_gate, run_tiny_training
run_gpu_correctness_gate(Path("experiments/competition_native_jax/gpu_correctness"))
run_tiny_training(Path("experiments/competition_native_jax/tiny"), max_transitions=256, max_updates=1)
print("R-E tiny complete")
PY
