#!/usr/bin/env bash
# Stage 3A parity + unit tests for E2E rebuild
set -euo pipefail
REPO_ROOT="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${REPO_ROOT}/third_party/generals-bots"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="${HOME}/quantsilico-runtime/jax_cache"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

echo "=== Quick import/reset smoke ==="
python - <<'PY'
import jax
from generals_bot.competition_native_jax.competition_env_jax import reset_one_jax, step_one_jax, legal_mask_one_jax
import jax.numpy as jnp
key = jax.random.PRNGKey(0)
st = reset_one_jax(key, 21, 21)
print("state", st.armies.shape, "time", int(st.time))
m = legal_mask_one_jax(st, 0)
print("legal", int(m.sum()), "pass", bool(m[0]))
pass_a = jnp.array([[1,0,0,0,0],[1,0,0,0,0]], dtype=jnp.int32)
ns, r, t, tr, info = step_one_jax(st, pass_a)
print("step ok", int(ns.time), float(r[0]), bool(t))
PY

echo "=== pytest competition_native_jax (incl 3A) ==="
pytest -q tests/competition_native_jax -o cache_dir=/tmp/qs_pytest_cache --tb=short 2>&1 | tee experiments/manifests/_e2e_3a_pytest.log
echo "PYTEST_DONE"
