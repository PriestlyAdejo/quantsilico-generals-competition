#!/usr/bin/env python3
"""Cloud parent compatibility + clean 32x32 A100 baseline (CLOUD_GPU_LAST_PUSH_V1)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

# Must be set before JAX initializes the CUDA allocator.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")

import jax
import jax.numpy as jnp

from generals_bot.competition_native_jax.competition_env_jax import build_competition_reset_pool
from generals_bot.competition_native_jax.obs_memory import N_GLOBAL, N_SPATIAL
from generals_bot.competition_native_jax.transformer_jax import forward, init_params
from train.competition_native_jax.ema_jax import ema_update
from train.competition_native_jax.gae_jax import gae_advantages_batch_jit
from train.competition_native_jax.ppo_jax import make_optimizer, ppo_update
from train.competition_native_jax.rollout_selfplay_jax import collect_selfplay_batch
from train.competition_native_jax.train_jax import (
    _rss_bytes,
    _vram_used_mib,
    detect_jax_device,
    lineage_hashes,
    load_training_checkpoint,
    save_tree,
)

EXPECTED_LEARNER = "2b10b1e326ba4f3b6532441b6a9f11fbb696e9d90684c81d6105f893df12ece2"
PARENT = Path(
    os.environ.get(
        "CLOUD_PARENT",
        "/workspace/quantsilico-runtime/cloud_gpu_last_push_v1/parent_u1524",
    )
)
OUT = Path(
    os.environ.get(
        "CLOUD_RUNTIME",
        "/workspace/quantsilico-runtime/cloud_gpu_last_push_v1",
    )
)
NUM_ENVS = 32
ROLLOUT_LEN = 32
RESET_POOL = 4096
BASELINE_UPDATES = int(os.environ.get("CLOUD_BASELINE_UPDATES", "20"))


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _gpu_util_snapshot() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "gpu_util_pct": float(parts[0]),
            "mem_util_pct": float(parts[1]),
            "vram_used_mib": float(parts[2]),
            "vram_total_mib": float(parts[3]),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _cpu_util_pct() -> float | None:
    try:
        import psutil

        return float(psutil.cpu_percent(interval=0.5))
    except Exception:
        return None


def _one_update(params, opt_state, optimizer, ema, *, num_envs, rollout_len, reset_pool, seed):
    batch = collect_selfplay_batch(
        params,
        num_envs=num_envs,
        rollout_len=rollout_len,
        seed=seed,
        reset_pool_size=RESET_POOL,
        pool=reset_pool,
    )
    T, N = batch["rewards"].shape
    flat = {
        "spatial": batch["spatial"].reshape(T * N, *batch["spatial"].shape[2:]),
        "global": batch["global"].reshape(T * N, *batch["global"].shape[2:]),
        "mask": batch["mask"].reshape(T * N, -1),
        "actions": batch["actions"].reshape(T * N),
        "old_logp": batch["old_logp"].reshape(T * N),
    }
    vals = jnp.concatenate([batch["values"], batch["bootstrap_values"][None, :]], axis=0)
    adv, ret = gae_advantages_batch_jit(batch["rewards"], vals, batch["dones"])
    flat["advantages"] = adv.reshape(T * N)
    flat["returns"] = ret.reshape(T * N)
    params, opt_state, metrics = ppo_update(params, opt_state, optimizer, flat)
    ema = ema_update(ema, params)
    jax.block_until_ready(jax.tree_util.tree_leaves(params)[0])
    return params, opt_state, ema, metrics, T * N


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    smoke_dir = OUT / "compat_smoke"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    smoke_dir.mkdir(parents=True)

    device = detect_jax_device()
    lin = lineage_hashes()
    print("DEVICE", json.dumps(device), flush=True)
    print("LINEAGE", json.dumps(lin), flush=True)
    code_learner = lin.get("learner_implementation_hash")
    learner_drift = code_learner != EXPECTED_LEARNER
    if learner_drift:
        print(
            "WARN_LEARNER_HASH_DRIFT",
            {"code": code_learner, "parent_expected": EXPECTED_LEARNER},
            flush=True,
        )

    key = jax.random.PRNGKey(0)
    params_like = jax.device_put(init_params(key))
    _ = forward(params_like, jnp.zeros((N_SPATIAL, 21, 21)), jnp.zeros((N_GLOBAL,)))
    optimizer = make_optimizer(3e-4)
    opt_like = optimizer.init(params_like)

    loaded = load_training_checkpoint(PARENT, params_like=params_like, opt_state_like=opt_like)
    params = jax.device_put(loaded["params"])
    ema = jax.device_put(loaded["ema"])
    opt_state = loaded["opt_state"]
    meta0 = loaded["meta"]
    parent_learner = (meta0.get("lineage") or {}).get("learner_implementation_hash")
    if parent_learner != EXPECTED_LEARNER:
        rep = {
            "status": "CLOUD_PARENT_COMPATIBILITY_FAIL",
            "reason": "learner_mismatch",
            "parent_learner": parent_learner,
            "expected": EXPECTED_LEARNER,
        }
        _write(OUT / "cloud_parent_compatibility.json", rep)
        print(json.dumps(rep, indent=2))
        return 2
    if int(meta0.get("update", -1)) != 1524 or int(meta0.get("transitions", -1)) != 1_560_576:
        rep = {
            "status": "CLOUD_PARENT_COMPATIBILITY_FAIL",
            "reason": "update_transitions_mismatch",
            "update": meta0.get("update"),
            "transitions": meta0.get("transitions"),
        }
        _write(OUT / "cloud_parent_compatibility.json", rep)
        print(json.dumps(rep, indent=2))
        return 2

    key_pool = jax.random.PRNGKey(int(meta0.get("reset_pool_seed") or 7))
    reset_pool = build_competition_reset_pool(key_pool, RESET_POOL)
    jax.block_until_ready(jax.tree_util.tree_leaves(reset_pool)[0])

    # Warm + one update from parent
    params, opt_state, ema, m1, n1 = _one_update(
        params,
        opt_state,
        optimizer,
        ema,
        num_envs=NUM_ENVS,
        rollout_len=ROLLOUT_LEN,
        reset_pool=reset_pool,
        seed=11,
    )

    # Save smoke ckpt and reload
    save_tree(smoke_dir / "raw.npz", params)
    save_tree(smoke_dir / "ema.npz", ema)
    save_tree(smoke_dir / "opt_state.npz", opt_state)
    meta_smoke = {
        **meta0,
        "update": int(meta0["update"]) + 1,
        "transitions": int(meta0["transitions"]) + int(n1),
        "parent_class": "CLOUD_COMPAT_SMOKE_FROM_U1524",
    }
    (smoke_dir / "meta.json").write_text(json.dumps(meta_smoke, indent=2) + "\n", encoding="utf-8")
    (smoke_dir / "COMPLETE").write_text('{"ok": true}\n', encoding="utf-8")

    reloaded = load_training_checkpoint(smoke_dir, params_like=params_like, opt_state_like=opt_like)
    params2 = jax.device_put(reloaded["params"])
    ema2 = jax.device_put(reloaded["ema"])
    opt2 = reloaded["opt_state"]
    params2, opt2, ema2, m2, n2 = _one_update(
        params2,
        opt2,
        optimizer,
        ema2,
        num_envs=NUM_ENVS,
        rollout_len=ROLLOUT_LEN,
        reset_pool=reset_pool,
        seed=12,
    )

    compat = {
        "schema_version": 1,
        "kind": "CLOUD_PARENT_COMPATIBILITY",
        "status": "CLOUD_PARENT_COMPATIBILITY_PASS",
        "parent": str(PARENT),
        "update": 1524,
        "transitions": 1_560_576,
        "parent_learner_implementation_hash": EXPECTED_LEARNER,
        "code_learner_implementation_hash": code_learner,
        "learner_hash_drift": bool(learner_drift),
        "learner_hash_note": (
            "Parent checkpoint lineage is 2b10…; current worktree hashes differently. "
            "Frozen 2b10 source bytes were not recoverable from git/history. "
            "Compat PASS means load→update→reload→update succeeded on A100."
        ),
        "device": device,
        "lineage_code": lin,
        "smoke_update_metrics": {k: float(v) for k, v in m1.items()},
        "reload_update_metrics": {k: float(v) for k, v in m2.items()},
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "xla_mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        "written_at": datetime.now(UTC).isoformat(),
    }
    _write(OUT / "cloud_parent_compatibility.json", compat)
    print("CLOUD_PARENT_COMPATIBILITY_PASS", flush=True)

    # Clean 32x32 baseline: fresh load of IMMUTABLE parent, warm, then timed updates only
    loaded = load_training_checkpoint(PARENT, params_like=params_like, opt_state_like=opt_like)
    params = jax.device_put(loaded["params"])
    ema = jax.device_put(loaded["ema"])
    opt_state = loaded["opt_state"]
    key_pool = jax.random.PRNGKey(12345)
    reset_pool = build_competition_reset_pool(key_pool, RESET_POOL)
    jax.block_until_ready(jax.tree_util.tree_leaves(reset_pool)[0])
    # Warm (excluded from TPS)
    params, opt_state, ema, _, _ = _one_update(
        params,
        opt_state,
        optimizer,
        ema,
        num_envs=NUM_ENVS,
        rollout_len=ROLLOUT_LEN,
        reset_pool=reset_pool,
        seed=100,
    )
    jax.block_until_ready(jax.tree_util.tree_leaves(params)[0])

    transitions = 0
    updates = 0
    last_metrics = {}
    t0 = time.perf_counter()
    for i in range(BASELINE_UPDATES):
        params, opt_state, ema, last_metrics, n = _one_update(
            params,
            opt_state,
            optimizer,
            ema,
            num_envs=NUM_ENVS,
            rollout_len=ROLLOUT_LEN,
            reset_pool=reset_pool,
            seed=200 + i,
        )
        transitions += int(n)
        updates += 1
    # Async-correct: block before stopping the clock
    jax.block_until_ready(jax.tree_util.tree_leaves(params)[0])
    elapsed = time.perf_counter() - t0
    tps = transitions / max(elapsed, 1e-6)
    snap = _gpu_util_snapshot()
    cpu = _cpu_util_pct()
    baseline = {
        "schema_version": 1,
        "kind": "CLOUD_A100_BASELINE_32X32",
        "status": "RECORDED",
        "num_envs": NUM_ENVS,
        "rollout_len": ROLLOUT_LEN,
        "reset_pool_size": RESET_POOL,
        "updates": updates,
        "transitions": transitions,
        "elapsed_s": elapsed,
        "valid_learning_tps": tps,
        "measured_tps": tps,
        "gpu": snap,
        "cpu_util_pct": cpu,
        "peak_vram_mib_train_api": _vram_used_mib(),
        "host_rss_bytes": _rss_bytes(),
        "device": device,
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "parent": str(PARENT),
        "last_metrics": {k: float(v) for k, v in last_metrics.items()} if last_metrics else {},
        "written_at": datetime.now(UTC).isoformat(),
        "contamination_note": "canary/dashboard must not run during this measurement",
    }
    _write(OUT / "cloud_a100_baseline_32x32.json", baseline)
    # also mirror into repo manifests if present
    repo_man = Path("/workspace/quantsilico-generals/experiments/manifests")
    if repo_man.is_dir():
        _write(repo_man / "cloud_a100_baseline_32x32.json", baseline)
        _write(repo_man / "cloud_parent_compatibility.json", compat)

    print(
        json.dumps(
            {
                "CLOUD_PARENT_COMPATIBILITY_PASS": True,
                "tps": tps,
                "gpu_util": snap.get("gpu_util_pct"),
                "vram_used_mib": snap.get("vram_used_mib"),
                "cpu_util_pct": cpu,
                "jax": jax.__version__,
                "backend": jax.default_backend(),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
