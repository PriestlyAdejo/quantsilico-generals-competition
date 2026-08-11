#!/usr/bin/env bash
set -euo pipefail
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
REPO="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}:${REPO}/third_party/generals-bots"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python <<'PY'
import json
from train.competition_native_jax.train_jax import lineage_hashes, detect_jax_device
print(json.dumps({"lineage": lineage_hashes(), "device": detect_jax_device()}, indent=2))
PY
echo "COMMIT=$(git rev-parse HEAD)"
echo "BRANCH=$(git branch --show-current)"
pgrep -af '_run_v4_1|train_jax' || echo NO_TRAINER
