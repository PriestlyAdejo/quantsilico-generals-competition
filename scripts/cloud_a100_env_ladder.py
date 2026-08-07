#!/usr/bin/env python3
"""A100 env scaling ladder after clean 32x32 baseline. Times only post-warm updates."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
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
    _vram_used_mib,
    detect_jax_device,
    load_training_checkpoint,
)

PARENT = Path(
    os.environ.get(
        "CLOUD_PARENT", "/workspace/quantsilico-runtime/cloud_gpu_last_push_v1/parent_u1524"
    )
)
OUT = Path(os.environ.get("CLOUD_RUNTIME", "/workspace/quantsilico-runtime/cloud_gpu_last_push_v1"))
RESET_POOL = 4096
TIMED_UPDATES = int(os.environ.get("CLOUD_LADDER_UPDATES", "10"))
# ladder: keep rollout_len=32 first, scale envs; then a couple geometry probes
LADDER = [
    {"num_envs": 32, "rollout_len": 32, "tag": "e32_r32"},
    {"num_envs": 64, "rollout_len": 32, "tag": "e64_r32"},
    {"num_envs": 128, "rollout_len": 32, "tag": "e128_r32"},
    {"num_envs": 256, "rollout_len": 32, "tag": "e256_r32"},
    {"num_envs": 512, "rollout_len": 32, "tag": "e512_r32"},
    {"num_envs": 64, "rollout_len": 64, "tag": "e64_r64"},
    {"num_envs": 128, "rollout_len": 64, "tag": "e128_r64"},
]


def _gpu_snap() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "gpu_util_pct": float(parts[0]),
            "vram_used_mib": float(parts[1]),
            "vram_total_mib": float(parts[2]),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


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


def run_cfg(cfg: dict, *, params_like, opt_like, reset_pool) -> dict:
    loaded = load_training_checkpoint(PARENT, params_like=params_like, opt_state_like=opt_like)
    params = jax.device_put(loaded["params"])
    ema = jax.device_put(loaded["ema"])
    opt_state = loaded["opt_state"]
    optimizer = make_optimizer(3e-4)
    ne, rl = int(cfg["num_envs"]), int(cfg["rollout_len"])
    print(f"LADDER_START {cfg['tag']}", flush=True)
    try:
        # warm excluded
        params, opt_state, ema, _, _ = _one_update(
            params,
            opt_state,
            optimizer,
            ema,
            num_envs=ne,
            rollout_len=rl,
            reset_pool=reset_pool,
            seed=1,
        )
        transitions = 0
        t0 = time.perf_counter()
        last = {}
        for i in range(TIMED_UPDATES):
            params, opt_state, ema, last, n = _one_update(
                params,
                opt_state,
                optimizer,
                ema,
                num_envs=ne,
                rollout_len=rl,
                reset_pool=reset_pool,
                seed=10 + i,
            )
            transitions += int(n)
        jax.block_until_ready(jax.tree_util.tree_leaves(params)[0])
        elapsed = time.perf_counter() - t0
        tps = transitions / max(elapsed, 1e-6)
        snap = _gpu_snap()
        row = {
            **cfg,
            "status": "OK",
            "updates": TIMED_UPDATES,
            "transitions": transitions,
            "elapsed_s": elapsed,
            "valid_learning_tps": tps,
            "gpu": snap,
            "vram_api_mib": _vram_used_mib(),
            "last_metrics": {k: float(v) for k, v in last.items()} if last else {},
        }
        print(f"LADDER_DONE {cfg['tag']} tps={tps:.2f} gpu={snap}", flush=True)
        return row
    except Exception as e:  # noqa: BLE001
        row = {
            **cfg,
            "status": "ERROR",
            "error": str(e),
            "valid_learning_tps": 0.0,
            "gpu": _gpu_snap(),
        }
        print(f"LADDER_ERR {cfg['tag']} {e}", flush=True)
        return row


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = detect_jax_device()
    key = jax.random.PRNGKey(0)
    params_like = jax.device_put(init_params(key))
    _ = forward(params_like, jnp.zeros((N_SPATIAL, 21, 21)), jnp.zeros((N_GLOBAL,)))
    optimizer = make_optimizer(3e-4)
    opt_like = optimizer.init(params_like)
    print("BUILD_POOL", RESET_POOL, flush=True)
    t_pool = time.perf_counter()
    reset_pool = build_competition_reset_pool(jax.random.PRNGKey(7), RESET_POOL)
    jax.block_until_ready(jax.tree_util.tree_leaves(reset_pool)[0])
    pool_s = time.perf_counter() - t_pool
    print(f"POOL_READY {pool_s:.1f}s", flush=True)

    rows = []
    for cfg in LADDER:
        rows.append(run_cfg(cfg, params_like=params_like, opt_like=opt_like, reset_pool=reset_pool))
        # persist partial after each rung
        rep = {
            "schema_version": 1,
            "kind": "CLOUD_A100_ENV_LADDER",
            "status": "PARTIAL"
            if any(r.get("status") != "OK" for r in rows) or len(rows) < len(LADDER)
            else "COMPLETE",
            "device": device,
            "reset_pool_size": RESET_POOL,
            "pool_build_s": pool_s,
            "timed_updates": TIMED_UPDATES,
            "parent": str(PARENT),
            "rows": rows,
            "written_at": datetime.now(UTC).isoformat(),
        }
        ok = [r for r in rows if r.get("status") == "OK"]
        if ok:
            best = max(ok, key=lambda r: float(r["valid_learning_tps"]))
            rep["best"] = best
        (OUT / "cloud_a100_env_ladder.json").write_text(json.dumps(rep, indent=2) + "\n")

    rep["status"] = "COMPLETE"
    (OUT / "cloud_a100_env_ladder.json").write_text(json.dumps(rep, indent=2) + "\n")
    man = Path("/workspace/quantsilico-generals/experiments/manifests")
    if man.is_dir():
        (man / "cloud_a100_env_ladder.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(
        json.dumps(
            {"best": rep.get("best"), "n_ok": len([r for r in rows if r.get("status") == "OK"])},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
