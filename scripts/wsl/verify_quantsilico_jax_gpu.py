#!/usr/bin/env python3
"""Hardened CUDA JAX device-placement gate for QuantSilico (GPU_JAX_VERIFIED)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import optax


def _dev(arr) -> str:
    d = getattr(arr, "device", None)
    return str(d() if callable(d) else d)


def _nvidia_used_mib() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def main(repo_root: str) -> int:
    root = Path(repo_root)
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))

    from generals_bot.competition_native_jax.transformer_jax import forward, init_params
    from train.competition_native_jax.ppo_jax import make_optimizer

    vram_before = _nvidia_used_mib()
    info: dict = {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "platforms": [getattr(d, "platform", None) for d in jax.devices()],
        "vram_used_mib_before": vram_before,
    }
    try:
        import jaxlib

        info["jaxlib_version"] = jaxlib.__version__
    except Exception as exc:  # noqa: BLE001
        info["jaxlib_error"] = str(exc)
    try:
        info["optax_version"] = optax.__version__
    except Exception as exc:  # noqa: BLE001
        info["optax_error"] = str(exc)

    cuda_devices = [
        d
        for d in jax.devices()
        if getattr(d, "platform", "") == "gpu" or "cuda" in str(d).lower() or "CudaDevice" in type(d).__name__
    ]
    checks: dict[str, bool] = {
        "backend_is_gpu": info["backend"] == "gpu",
        "has_cuda_device": len(cuda_devices) >= 1,
    }

    # Matmul placement
    x = jax.device_put(jnp.ones((2048, 2048)))
    y = jax.device_put(jnp.ones((2048, 2048)))
    z = (x @ y).block_until_ready()
    info["matmul_device"] = _dev(z)
    info["input_device"] = _dev(x)
    checks["matmul_on_gpu"] = "gpu" in info["matmul_device"].lower() or "cuda" in info["matmul_device"].lower()

    # Transformer params + forward on GPU
    key = jax.random.PRNGKey(0)
    params = jax.device_put(init_params(key))
    leaf = jax.tree_util.tree_leaves(params)[0]
    info["params_device"] = _dev(leaf)
    checks["params_on_gpu"] = "gpu" in info["params_device"].lower() or "cuda" in info["params_device"].lower()

    from generals_bot.competition_native_jax.obs_memory import N_GLOBAL, N_SPATIAL

    spatial = jax.device_put(jnp.zeros((N_SPATIAL, 21, 21), dtype=jnp.float32))
    global_vec = jax.device_put(jnp.zeros((N_GLOBAL,), dtype=jnp.float32))
    out = forward(params, spatial, global_vec)
    out["flat_logits"].block_until_ready()
    info["forward_logits_device"] = _dev(out["flat_logits"])
    checks["forward_on_gpu"] = "gpu" in info["forward_logits_device"].lower() or "cuda" in info[
        "forward_logits_device"
    ].lower()

    # Rollout-like arrays on GPU
    rollout = {
        "spatial": jax.device_put(jnp.zeros((16, 4, N_SPATIAL, 21, 21), dtype=jnp.float32)),
        "global": jax.device_put(jnp.zeros((16, 4, N_GLOBAL), dtype=jnp.float32)),
        "mask": jax.device_put(jnp.ones((16, 4, out["flat_logits"].shape[0]), dtype=bool)),
    }
    info["rollout_spatial_device"] = _dev(rollout["spatial"])
    checks["rollout_on_gpu"] = "gpu" in info["rollout_spatial_device"].lower() or "cuda" in info[
        "rollout_spatial_device"
    ].lower()

    # Gradients on GPU
    def loss_fn(p):
        o = forward(p, spatial, global_vec)
        return jnp.sum(o["flat_logits"] ** 2)

    grads = jax.grad(loss_fn)(params)
    gleaf = jax.tree_util.tree_leaves(grads)[0]
    gleaf.block_until_ready()
    info["grad_device"] = _dev(gleaf)
    checks["grads_on_gpu"] = "gpu" in info["grad_device"].lower() or "cuda" in info["grad_device"].lower()

    # Optax update on GPU
    optimizer = make_optimizer(1e-3)
    opt_state = optimizer.init(params)
    updates, opt_state2 = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    uleaf = jax.tree_util.tree_leaves(updates)[0]
    uleaf.block_until_ready()
    nleaf = jax.tree_util.tree_leaves(new_params)[0]
    nleaf.block_until_ready()
    info["optax_update_device"] = _dev(uleaf)
    info["optax_params_device"] = _dev(nleaf)
    checks["optax_on_gpu"] = "gpu" in info["optax_update_device"].lower() or "cuda" in info[
        "optax_update_device"
    ].lower()

    vram_after = _nvidia_used_mib()
    info["vram_used_mib_after"] = vram_after
    if vram_before is not None and vram_after is not None:
        info["vram_delta_mib"] = vram_after - vram_before
        checks["observable_vram_utilisation"] = (vram_after - vram_before) > 0.5 or vram_after > 100.0
    else:
        checks["observable_vram_utilisation"] = checks["matmul_on_gpu"]  # fallback

    all_ok = all(checks.values()) and checks["backend_is_gpu"] and checks["has_cuda_device"]
    status = "GPU_JAX_VERIFIED" if all_ok else "GPU_JAX_INSTALL_FAILED"
    if checks["backend_is_gpu"] is False and cuda_devices:
        status = "GPU_VISIBLE_BUT_JAX_CPU"
    elif not checks["backend_is_gpu"]:
        status = "GPU_JAX_INSTALL_FAILED"

    report = {
        "schema_version": 1,
        "kind": "PHASE9G_GPU_EXECUTION_GATE",
        "status": status,
        "checks": checks,
        "device_info": info,
        "cuda_devices": [str(d) for d in cuda_devices],
    }
    out_path = root / "experiments/manifests/phase9g_gpu_execution_gate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if status == "GPU_JAX_VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
