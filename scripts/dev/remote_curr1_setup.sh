#!/usr/bin/env bash
# SPAWN-DISTANCE-CURRICULUM-R1 pod setup (run ON the pod via ssh).
# Idempotent: clone repo branch + engine submodule, uv venv 3.12, CUDA JAX.
set -euo pipefail

REPO_DIR=/workspace/repo
BRANCH=research/phase9g-competition-native-jax-preovernight-v1
CLONE_URL=https://github.com/PriestlyAdejo/quantsilico-generals-competition.git

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --branch "$BRANCH" --depth 1 "$CLONE_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git submodule update --init --depth 1 third_party/generals-bots

if [ ! -x /root/.local/bin/uv ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="/root/.local/bin:$PATH"

if [ ! -x "$REPO_DIR/.venv312/bin/python" ]; then
  uv venv "$REPO_DIR/.venv312" --python 3.12
fi
uv pip install --python "$REPO_DIR/.venv312/bin/python" \
  "jax[cuda12]==0.11.0" "optax==0.2.8" pyyaml numpy

echo "=== setup done ==="
"$REPO_DIR/.venv312/bin/python" -c "import jax; print('JAX', jax.__version__, jax.default_backend(), jax.devices())"
