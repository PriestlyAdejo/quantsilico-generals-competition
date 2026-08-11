"""Hot-path component profiler for competition_native_jax remediation."""
from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def main() -> None:
    from generals import GeneralsEnv
    from generals.core import game
    from generals_bot.competition_native_jax.transformer_jax import forward, init_params
    from train.competition_native_jax.rollout_selfplay_jax import (
        _batch_obs_from_states,
        _obs_to_spatial_global_np,
        collect_selfplay_batch,
        forward_batch,
        legal_mask_vectorised,
    )

    key = jax.random.PRNGKey(0)
    params = init_params(key)
    env = GeneralsEnv(mode="competition", pool_size=64)
    key, pk = jax.random.split(key)
    pool, _ = env.reset(pk)
    n = 8
    states = jax.vmap(env.init_state)(jax.random.split(key, n))
    times: dict[str, float] = {}

    # 1 env transition (batched)
    pass_a = jnp.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=jnp.int32)
    actions = jnp.tile(pass_a[None, :, :], (n, 1, 1))
    step_fn = lambda s, a: env.step(s, a, pool)
    step_batch = jax.jit(jax.vmap(step_fn))
    t0 = time.perf_counter()
    ts, states2 = step_batch(states, actions)
    jax.block_until_ready(states2)
    times["env_step_batch_compile_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(20):
        ts, states2 = step_batch(states2, actions)
        jax.block_until_ready(states2)
    times["env_step_batch_20_s"] = time.perf_counter() - t0
    times["env_transition_per_env_ms"] = 1000.0 * times["env_step_batch_20_s"] / (20 * n)

    # obs+mask host batch
    t0 = time.perf_counter()
    for _ in range(10):
        sp, gv, m = _batch_obs_from_states(states2, 0)
    times["obs_mask_batch_10_s"] = time.perf_counter() - t0
    times["obs_mask_per_env_ms"] = 1000.0 * times["obs_mask_batch_10_s"] / (10 * n)

    # single forward
    sp0 = jnp.asarray(sp[0])
    gv0 = jnp.asarray(gv[0])
    t0 = time.perf_counter()
    out = forward(params, sp0, gv0)
    out["flat_logits"].block_until_ready()
    times["policy_forward_single_compile_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(20):
        out = forward(params, sp0, gv0)
        out["flat_logits"].block_until_ready()
    times["policy_forward_single_20_s"] = time.perf_counter() - t0

    # batched forward
    t0 = time.perf_counter()
    bout = forward_batch(params, jnp.asarray(sp), jnp.asarray(gv))
    bout["flat_logits"].block_until_ready()
    times["policy_forward_batch_compile_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(20):
        bout = forward_batch(params, jnp.asarray(sp), jnp.asarray(gv))
        bout["flat_logits"].block_until_ready()
    times["policy_forward_batch_20_s"] = time.perf_counter() - t0
    times["policy_batch_speedup_vs_naive"] = (times["policy_forward_single_20_s"] * n) / max(
        times["policy_forward_batch_20_s"], 1e-9
    )

    # full rollout collect (2 steps)
    t0 = time.perf_counter()
    batch = collect_selfplay_batch(params, num_envs=n, rollout_len=2, seed=1)
    jax.block_until_ready(batch["rewards"])
    times["full_rollout_2steps_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    batch = collect_selfplay_batch(params, num_envs=n, rollout_len=8, seed=2)
    jax.block_until_ready(batch["rewards"])
    times["full_rollout_8steps_s"] = time.perf_counter() - t0
    times["full_rollout_tps_estimate"] = (n * 8) / max(times["full_rollout_8steps_s"], 1e-9)

    total = times["full_rollout_8steps_s"]
    pct = {
        "env_step_share_pct": 100.0 * (times["env_step_batch_20_s"] / 20 * 8) / max(total, 1e-9),
        "obs_mask_share_pct": 100.0 * (times["obs_mask_batch_10_s"] / 10 * 8) / max(total, 1e-9),
        "policy_batch_share_pct": 100.0 * (times["policy_forward_batch_20_s"] / 20 * 8) / max(total, 1e-9),
    }
    # clamp explanatory shares
    for k, v in list(pct.items()):
        pct[k] = float(min(v, 100.0))

    report = {
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_HOT_PATH_PROFILE",
        "num_envs": n,
        "times_s": times,
        "approximate_shares_pct": pct,
        "rollout_architecture": "END_TO_END_OFFICIAL_JAX_ROLLOUT",
        "baseline_valid_tps": [0.1513, 0.185],
        "notes": [
            "Shares extrapolated from microbenchmarks onto full_rollout_8steps wall time",
            "Obs/mask still host Python loop over envs calling official get_observation",
            "Env step and policy forward are jax.vmap+jit batched",
        ],
    }
    Path("experiments/manifests").mkdir(parents=True, exist_ok=True)
    Path("experiments/reports").mkdir(parents=True, exist_ok=True)
    Path("experiments/manifests/competition_native_jax_hot_path_profile.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# Hot-path profile",
        "",
        f"Architecture: `{report['rollout_architecture']}`",
        "",
        "## Measured times",
        "",
    ]
    for k, v in times.items():
        md.append(f"- `{k}`: {v:.6f}")
    md.extend(["", "## Approximate shares of 8-step rollout", ""])
    for k, v in pct.items():
        md.append(f"- `{k}`: {v:.1f}%")
    md.extend(
        [
            "",
            f"Full-rollout TPS estimate (8 steps × {n} envs): **{times['full_rollout_tps_estimate']:.3f}**",
            "",
            f"Baseline complete-loop TPS: 0.151–0.185",
            "",
        ]
    )
    Path("experiments/reports/competition_native_jax_hot_path_profile.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
