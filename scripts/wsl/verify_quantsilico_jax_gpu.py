#!/usr/bin/env python3
"""Verify CUDA JAX device placement for QuantSilico."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp


def main(repo_root: str) -> int:
    root = Path(repo_root)
    info = {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "platforms": [getattr(d, "platform", None) for d in jax.devices()],
    }
    try:
        import jaxlib

        info["jaxlib_version"] = jaxlib.__version__
    except Exception as exc:  # noqa: BLE001
        info["jaxlib_error"] = str(exc)

    gpu_ok = info["backend"] == "gpu" and any(p == "gpu" for p in info["platforms"])
    status = "GPU_JAX_VERIFIED" if gpu_ok else "GPU_VISIBLE_BUT_JAX_CPU"
    if not any("cuda" in str(d).lower() or "gpu" in str(d).lower() for d in jax.devices()) and info["backend"] != "gpu":
        status = "GPU_JAX_INSTALL_FAILED"

    # Device placement proof
    x = jax.device_put(jnp.ones((1024, 1024)))
    y = jax.device_put(jnp.ones((1024, 1024)))
    z = (x @ y).block_until_ready()
    def _dev(arr) -> str:
        d = getattr(arr, "device", None)
        return str(d() if callable(d) else d)

    info["matmul_device"] = _dev(z)
    info["input_device"] = _dev(x)

    def loss(w):
        return jnp.sum((w @ x) ** 2)

    w = jax.device_put(jnp.ones((1024, 1024)))
    g = jax.grad(loss)(w).block_until_ready()
    info["grad_device"] = _dev(g)

    report = {
        "schema_version": 1,
        "kind": "PHASE9G_GPU_EXECUTION_GATE",
        "status": status,
        "device_info": info,
    }
    out = root / "experiments/manifests/phase9g_gpu_execution_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0 if status == "GPU_JAX_VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
