#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition
sed -i 's/\r$//' scripts/wsl/_run_v4_1_pipeline.sh
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
export PYTHONPATH="src:.:third_party/generals-bots"
python - <<'PY'
from train.competition_native_jax.train_jax import lineage_hashes, env_implementation_hash
h = env_implementation_hash()
print("env_implementation_hash", h)
print("expected_prefix", "c6339411")
print("match", h.startswith("c6339411"))
print(lineage_hashes())
PY
