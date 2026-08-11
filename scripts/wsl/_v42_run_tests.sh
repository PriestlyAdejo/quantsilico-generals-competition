#!/usr/bin/env bash
set -euo pipefail
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
REPO="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
cd "$REPO"
export PYTHONPATH="${REPO}/src:${REPO}:${REPO}/third_party/generals-bots"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
pytest -q tests/competition_native_jax/test_v4_2_reset_pool.py tests/competition_native_jax/test_jax_core.py \
  -o cache_dir=/tmp/qs_pytest_cache --tb=short 2>&1 | tee /tmp/v42_tests.out
echo EXIT:$?
