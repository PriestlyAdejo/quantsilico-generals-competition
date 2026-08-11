#!/usr/bin/env bash
set -euxo pipefail
REPO_ROOT="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export PYTHONUNBUFFERED=1
python -u - <<'PY'
print("importing...", flush=True)
import jax
print("jax", jax.__version__, jax.default_backend(), jax.devices(), flush=True)
from generals_bot.competition_native_jax.transformer_jax import init_params
from train.competition_native_jax.rollout_selfplay_jax import collect_selfplay_batch
print("init params", flush=True)
params = init_params(jax.random.PRNGKey(0))
print("collect batch start", flush=True)
batch = collect_selfplay_batch(params, num_envs=1, rollout_len=2, seed=0)
print("batch keys", {k: getattr(v, "shape", None) for k, v in batch.items() if k != "backend"}, flush=True)
print("OK", flush=True)
PY
