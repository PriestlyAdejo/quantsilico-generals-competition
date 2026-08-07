#!/usr/bin/env bash
# Restore the verified Python 3.12/JAX CUDA environment after a container reset.
set -euo pipefail

BUNDLE_DIR="${1:-/workspace/.venvs/quantsilico-generals}"
ARCHIVE="${BUNDLE_DIR}/environment.tar.zst"

test -f "${ARCHIVE}"
test -f "${ARCHIVE}.sha256"
cd "${BUNDLE_DIR}"
sha256sum --check "$(basename "${ARCHIVE}").sha256"
tar -C / -I zstd -xf "${ARCHIVE}"
/tmp/qs-venv312/bin/python - <<'PY'
import jax

devices = jax.devices()
assert jax.__version__ == "0.11.0", jax.__version__
assert jax.default_backend() == "gpu", jax.default_backend()
assert len(devices) == 1, devices
assert "A100" in devices[0].device_kind, devices[0].device_kind
print(jax.__version__, jax.default_backend(), devices[0].device_kind)
PY
