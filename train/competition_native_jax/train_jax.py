"""Canonical JAX training entrypoints."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from generals_bot.competition_native_jax.transformer_jax import forward, init_params
from train.competition_native_jax.ema_jax import ema_update
from train.competition_native_jax.gae_jax import gae_advantages
from train.competition_native_jax.ppo_jax import assert_zero_update_ratio, make_optimizer, ppo_update
from train.competition_native_jax.rollout_selfplay_jax import collect_selfplay_batch


def detect_jax_device() -> dict[str, Any]:
    info: dict[str, Any] = {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "jax_gpu": any(getattr(d, "platform", "") == "gpu" or "cuda" in str(d).lower() for d in jax.devices()),
    }
    try:
        import jaxlib

        info["jaxlib_version"] = jaxlib.__version__
    except Exception:
        info["jaxlib_version"] = None
    return info


def run_gpu_correctness_gate(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = detect_jax_device()
    key = jax.random.PRNGKey(0)
    params = init_params(key)
    # Place on default device
    params = jax.device_put(params)
    spatial = jax.device_put(jnp.zeros((8, 21, 21), dtype=jnp.float32))
    global_vec = jax.device_put(jnp.zeros((8,), dtype=jnp.float32))
    out = forward(params, spatial, global_vec)
    out["flat_logits"].block_until_ready()
    mask = jnp.zeros(out["flat_logits"].shape[0], dtype=bool).at[0].set(True)
    mask = mask.at[1:20].set(True)
    rho = float(assert_zero_update_ratio(out["flat_logits"], mask, jnp.array(0)))
    # grad placement check
    def loss_fn(p):
        o = forward(p, spatial, global_vec)
        return jnp.sum(o["flat_logits"] ** 2)

    grads = jax.grad(loss_fn)(params)
    leaf = jax.tree_util.tree_leaves(grads)[0]
    report = {
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_GPU_CORRECTNESS_GATE",
        "status": "PASSED" if abs(rho - 1.0) < 1e-5 else "FAILED",
        "zero_update_rho": rho,
        "device": device,
        "params_on_device": str(jax.devices()[0]),
        "grad_device": str(leaf.device()) if hasattr(leaf, "device") else str(jax.devices()[0]),
        "note": "JAX forward+grad identity; GPU verification is separate gate.",
    }
    (out_dir / "gpu_correctness_gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_tiny_training(out_dir: Path, *, max_transitions: int = 512, max_updates: int = 2, seed: int = 0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    device = detect_jax_device()
    key = jax.random.PRNGKey(seed)
    params = init_params(key)
    ema = params
    optimizer = make_optimizer(3e-4)
    opt_state = optimizer.init(params)
    transitions = 0
    updates = 0
    while transitions < max_transitions and updates < max_updates:
        batch = collect_selfplay_batch(params, num_envs=2, rollout_len=16, seed=seed + updates)
        # Flatten [T,N,...] -> [T*N,...]
        T, N = batch["rewards"].shape
        flat = {
            "spatial": batch["spatial"].reshape(T * N, *batch["spatial"].shape[2:]),
            "global": batch["global"].reshape(T * N, *batch["global"].shape[2:]),
            "mask": batch["mask"].reshape(T * N, -1),
            "actions": batch["actions"].reshape(T * N),
            "old_logp": batch["old_logp"].reshape(T * N),
        }
        # GAE per env then flatten
        advs = []
        rets = []
        for i in range(N):
            vals = jnp.concatenate([batch["values"][:, i], batch["values"][-1:, i]])
            adv, ret = gae_advantages(batch["rewards"][:, i], vals, batch["dones"][:, i])
            advs.append(adv)
            rets.append(ret)
        flat["advantages"] = jnp.stack(advs, axis=1).reshape(T * N)
        flat["returns"] = jnp.stack(rets, axis=1).reshape(T * N)
        params, opt_state, metrics = ppo_update(params, opt_state, optimizer, flat)
        ema = ema_update(ema, params)
        transitions += T * N
        updates += 1
    elapsed = time.perf_counter() - t0
    # save npz via flattened leaves
    def save_tree(path: Path, tree):
        flat_tree, _ = jax.tree_util.tree_flatten_with_path(tree)
        np.savez_compressed(path, **{str(k): np.asarray(v) for k, v in flat_tree})

    raw_path = out_dir / "tiny_raw.npz"
    ema_path = out_dir / "tiny_ema.npz"
    save_tree(raw_path, params)
    save_tree(ema_path, ema)
    report = {
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_TINY_TRAIN",
        "status": "COMPLETED",
        "transitions": transitions,
        "updates": updates,
        "elapsed_s": elapsed,
        "measured_tps": transitions / max(elapsed, 1e-6),
        "device": device,
        "checkpoint_raw": str(raw_path).replace("\\", "/"),
        "checkpoint_ema": str(ema_path).replace("\\", "/"),
        "jax_gpu_used": bool(device.get("jax_gpu")),
    }
    (out_dir / "tiny_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
