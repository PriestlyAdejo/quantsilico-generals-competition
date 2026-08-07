"""V4.2 component profile (attribution-only) + GAE audit."""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from generals_bot.competition_native_jax.transformer_jax import init_params
from generals_bot.competition_native_jax.competition_env_jax import build_competition_reset_pool
from train.competition_native_jax.gae_jax import gae_advantages_batch, gae_advantages
from train.competition_native_jax.ppo_jax import make_optimizer, ppo_update
from train.competition_native_jax.rollout_selfplay_jax import collect_selfplay_batch
from train.competition_native_jax.train_jax import _vram_used_mib, detect_jax_device, lineage_hashes


def _time(fn, *args, **kwargs):
    out = fn(*args, **kwargs)
    if isinstance(out, (tuple, list)):
        for o in out:
            if hasattr(o, "block_until_ready"):
                o.block_until_ready()
            elif isinstance(o, dict):
                jax.block_until_ready(jax.tree_util.tree_leaves(o)[0])
        return out
    if isinstance(out, dict):
        jax.block_until_ready(out.get("rewards", jax.tree_util.tree_leaves(out)[0]))
        return out
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    return out


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    num_envs, rollout_len, pool_size = 32, 16, 4096
    key = jax.random.PRNGKey(0)
    params = init_params(key)
    opt = make_optimizer(3e-4)
    opt_state = opt.init(params)
    k_pool, _ = jax.random.split(key)
    pool = build_competition_reset_pool(k_pool, pool_size)
    jax.block_until_ready(jax.tree_util.tree_leaves(pool)[0])

    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    batch = _time(
        collect_selfplay_batch,
        params,
        num_envs=num_envs,
        rollout_len=rollout_len,
        seed=0,
        reset_pool_size=pool_size,
        pool=pool,
    )
    timings["complete_rollout_collect"] = time.perf_counter() - t0

    T, N = batch["rewards"].shape
    flat = {
        "spatial": batch["spatial"].reshape(T * N, *batch["spatial"].shape[2:]),
        "global": batch["global"].reshape(T * N, *batch["global"].shape[2:]),
        "mask": batch["mask"].reshape(T * N, -1),
        "actions": batch["actions"].reshape(T * N),
        "old_logp": batch["old_logp"].reshape(T * N),
    }
    vals = jnp.concatenate([batch["values"], batch["bootstrap_values"][None, :]], axis=0)

    t0 = time.perf_counter()
    adv, ret = gae_advantages_batch(batch["rewards"], vals, batch["dones"])
    jax.block_until_ready(adv)
    timings["gae_batch_scan"] = time.perf_counter() - t0

    for i in range(min(N, 2)):
        a_i, r_i = gae_advantages(batch["rewards"][:, i], vals[:, i], batch["dones"][:, i])
        assert jnp.allclose(adv[:, i], a_i, atol=1e-5)

    flat["advantages"] = adv.reshape(T * N)
    flat["returns"] = ret.reshape(T * N)

    t0 = time.perf_counter()
    params, opt_state, metrics = ppo_update(params, opt_state, opt, flat)
    jax.block_until_ready(jax.tree_util.tree_leaves(params)[0])
    timings["ppo_full_batch_update"] = time.perf_counter() - t0

    total = sum(timings.values())
    ranked = sorted(
        [{"component": k, "seconds": v, "pct": 100.0 * v / max(total, 1e-9)} for k, v in timings.items()],
        key=lambda r: -r["seconds"],
    )

    warm = collect_selfplay_batch(
        params, num_envs=num_envs, rollout_len=rollout_len, seed=1, reset_pool_size=pool_size, pool=pool
    )
    jax.block_until_ready(warm["rewards"])
    t0 = time.perf_counter()
    b2 = collect_selfplay_batch(
        params, num_envs=num_envs, rollout_len=rollout_len, seed=2, reset_pool_size=pool_size, pool=pool
    )
    T2, N2 = b2["rewards"].shape
    flat2 = {
        "spatial": b2["spatial"].reshape(T2 * N2, *b2["spatial"].shape[2:]),
        "global": b2["global"].reshape(T2 * N2, *b2["global"].shape[2:]),
        "mask": b2["mask"].reshape(T2 * N2, -1),
        "actions": b2["actions"].reshape(T2 * N2),
        "old_logp": b2["old_logp"].reshape(T2 * N2),
    }
    vals2 = jnp.concatenate([b2["values"], b2["bootstrap_values"][None, :]], axis=0)
    adv2, ret2 = gae_advantages_batch(b2["rewards"], vals2, b2["dones"])
    flat2["advantages"] = adv2.reshape(T2 * N2)
    flat2["returns"] = ret2.reshape(T2 * N2)
    params, opt_state, _ = ppo_update(params, opt_state, opt, flat2)
    jax.block_until_ready(jax.tree_util.tree_leaves(params)[0])
    clean_elapsed = time.perf_counter() - t0
    clean_tps = (T2 * N2) / max(clean_elapsed, 1e-9)

    report = {
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_V4_2_PROFILE",
        "num_envs": num_envs,
        "rollout_len": rollout_len,
        "reset_pool_size": pool_size,
        "ranked_components": ranked,
        "gae_device_resident_batched": True,
        "gae_python_env_loop": False,
        "gae_bootstrap": "post_scan_forward_value",
        "reset_path": "device_competition_reset_pool_reused",
        "clean_profiler_disabled_valid_learning_tps_one_update": clean_tps,
        "peak_vram_mib": _vram_used_mib(),
        "device": detect_jax_device(),
        "lineage": lineage_hashes(),
        "last_metrics": {k: float(v) for k, v in metrics.items()},
    }
    man = repo / "experiments/manifests/competition_native_jax_v4_2_profile.json"
    md = repo / "experiments/reports/competition_native_jax_v4_2_profile.md"
    man.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# V4.2 component profile",
        "",
        f"Clean one-update valid-learning TPS (profiler-disabled): **{clean_tps:.2f}**",
        "",
        "GAE: device-resident batched reverse `lax.scan` (no Python env loop).",
        "Reset: device competition reset pool reused across collects.",
        "",
        "| Component | seconds | % |",
        "|---|---:|---:|",
    ]
    for r in ranked:
        lines.append(f"| {r['component']} | {r['seconds']:.4f} | {r['pct']:.1f} |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"clean_tps": clean_tps, "top": ranked[0]}, indent=2))


if __name__ == "__main__":
    main()
