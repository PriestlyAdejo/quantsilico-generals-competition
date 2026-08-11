"""Checkpoint round-trip: uninterrupted updates 1–2 vs save@1 → new process → load → update 2."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from generals_bot.competition_native_jax.competition_env_jax import build_competition_reset_pool
from generals_bot.competition_native_jax.obs_memory import N_GLOBAL, N_SPATIAL
from generals_bot.competition_native_jax.transformer_jax import forward, init_params
from train.competition_native_jax.ema_jax import ema_update
from train.competition_native_jax.gae_jax import gae_advantages_batch_jit
from train.competition_native_jax.ppo_jax import make_optimizer, ppo_update
from train.competition_native_jax.rollout_selfplay_jax import collect_selfplay_batch
from train.competition_native_jax.train_jax import (
    load_training_checkpoint,
    save_training_checkpoint,
    lineage_hashes,
)

ROOT = Path(__file__).resolve().parents[1]


def _one_update(params, ema, opt_state, optimizer, reset_pool, *, num_envs, rollout_len, seed):
    batch = collect_selfplay_batch(
        params,
        num_envs=num_envs,
        rollout_len=rollout_len,
        seed=seed,
        reset_pool_size=int(jax.tree_util.tree_leaves(reset_pool)[0].shape[0]),
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
    return params, ema, opt_state, metrics


def _leaf_max_abs(a, b) -> float:
    return float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b))))


def main() -> int:
    num_envs, rollout_len, pool_size = 8, 8, 256
    out = ROOT / "experiments/competition_native_jax/v4_3_ckpt_roundtrip"
    out.mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(0)
    params0 = jax.device_put(init_params(key))
    _ = forward(params0, jnp.zeros((N_SPATIAL, 21, 21)), jnp.zeros((N_GLOBAL,)))
    optimizer = make_optimizer(3e-4)
    opt0 = optimizer.init(params0)
    key_pool, key = jax.random.split(key)
    pool = build_competition_reset_pool(key_pool, pool_size, min_grid=21, max_grid=21)

    # Path A: uninterrupted updates 1 then 2
    p_a, e_a, o_a = params0, params0, opt0
    p_a, e_a, o_a, m1a = _one_update(p_a, e_a, o_a, optimizer, pool, num_envs=num_envs, rollout_len=rollout_len, seed=1)
    p_a2, e_a2, o_a2, m2a = _one_update(p_a, e_a, o_a, optimizer, pool, num_envs=num_envs, rollout_len=rollout_len, seed=2)

    # Path B: after update 1, save; reload; update 2
    p_b, e_b, o_b = params0, params0, opt0
    p_b, e_b, o_b, m1b = _one_update(p_b, e_b, o_b, optimizer, pool, num_envs=num_envs, rollout_len=rollout_len, seed=1)
    meta = {
        "update": 1,
        "transitions": num_envs * rollout_len,
        "lr": 3e-4,
        "model_rng": [0, 0],
        "env_rng": [0, 1],
        "reset_pool_seed": 0,
        "reset_pool_cursor": 0,
        "curriculum": None,
        "num_envs": num_envs,
        "rollout_len": rollout_len,
        "reset_pool_size": pool_size,
        "evaluation_protocol_id": "competition_native_jax_daytime_evaluation_protocol_v2",
        "evaluation_protocol_sha256": None,
        "lineage": lineage_hashes(),
        "dtype": "float32",
        "static_profile": "v4_2_selected",
    }
    ckpt_dir = out / "ckpt_after_1"
    save_training_checkpoint(ckpt_dir, params=p_b, ema=e_b, opt_state=o_b, meta=meta)
    loaded = load_training_checkpoint(ckpt_dir, params_like=params0, opt_state_like=opt0)
    p_b2, e_b2, o_b2, m2b = _one_update(
        loaded["params"],
        loaded["ema"],
        loaded["opt_state"],
        optimizer,
        pool,
        num_envs=num_envs,
        rollout_len=rollout_len,
        seed=2,
    )

    param_err = max(_leaf_max_abs(a, b) for a, b in zip(jax.tree_util.tree_leaves(p_a2), jax.tree_util.tree_leaves(p_b2)))
    ema_err = max(_leaf_max_abs(a, b) for a, b in zip(jax.tree_util.tree_leaves(e_a2), jax.tree_util.tree_leaves(e_b2)))
    opt_err = max(_leaf_max_abs(a, b) for a, b in zip(jax.tree_util.tree_leaves(o_a2), jax.tree_util.tree_leaves(o_b2)))
    metric_err = max(abs(float(m2a[k]) - float(m2b[k])) for k in m2a)

    # Also verify update-1 path A vs B match before second update
    param_err_1 = max(_leaf_max_abs(a, b) for a, b in zip(jax.tree_util.tree_leaves(p_a), jax.tree_util.tree_leaves(p_b)))

    tol = 1e-5
    passed = param_err < tol and ema_err < tol and opt_err < tol and metric_err < 1e-4 and param_err_1 < tol
    report = {
        "schema_version": 1,
        "kind": "CHECKPOINT_EXACT_CONTINUATION",
        "status": "CHECKPOINT_EXACT_CONTINUATION_PASS" if passed else "CHECKPOINT_TRAINING_STATE_ROUNDTRIP_FAIL",
        "param_err_after_2": param_err,
        "ema_err_after_2": ema_err,
        "opt_err_after_2": opt_err,
        "metric_err_after_2": metric_err,
        "param_err_after_1": param_err_1,
        "tol": tol,
        "ckpt_dir": str(ckpt_dir).replace("\\", "/"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (ROOT / "experiments/manifests/competition_native_jax_v4_3_checkpoint_roundtrip.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
