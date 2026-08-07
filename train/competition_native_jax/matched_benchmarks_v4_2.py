"""V4.2 matched benchmarks A–E and environment-ceiling ladder."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from generals_bot.competition_native_jax.competition_env_jax import (
    build_competition_reset_pool,
    empty_memory,
    index_to_engine_action_batch,
    legal_mask_batch_p0,
    legal_mask_batch_p1,
    observe_batch_p0,
    observe_batch_p1,
    step_batch_jax,
)
from generals_bot.competition_native_jax.constants import PASS_INDEX
from generals_bot.competition_native_jax.transformer_jax import forward_batch, init_params
from train.competition_native_jax.gae_jax import gae_advantages_batch_jit
from train.competition_native_jax.ppo_jax import make_optimizer, ppo_update
from train.competition_native_jax.rollout_selfplay_jax import collect_selfplay_batch
from train.competition_native_jax.train_jax import (
    _rss_bytes,
    _train_loop,
    _vram_used_mib,
    detect_jax_device,
    lineage_hashes,
)


def _gpu_sample() -> dict[str, Any]:
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,temperature.gpu,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_power_cap",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "gpu_util_pct": float(parts[0]),
            "gpu_mem_util_pct": float(parts[1]),
            "vram_mib": float(parts[2]),
            "power_w": float(parts[3]),
            "sm_clock_mhz": float(parts[4]),
            "temp_c": float(parts[5]),
            "thermal_slowdown": parts[6],
            "power_cap": parts[7],
        }
    except Exception as e:
        return {"error": str(e)}


def _pass_actions(n: int) -> jnp.ndarray:
    idx = jnp.full((n,), PASS_INDEX, dtype=jnp.int32)
    eng = index_to_engine_action_batch(idx)
    return jnp.stack([eng, eng], axis=1)


def bench_env_transition(num_envs: int, steps: int, seed: int = 0) -> dict:
    key = jax.random.PRNGKey(seed)
    pool = build_competition_reset_pool(key, max(num_envs * 4, 64), min_grid=21, max_grid=21)
    states = jax.tree_util.tree_map(lambda x: x[:num_envs], pool)
    actions = _pass_actions(num_envs)

    def body(s, _):
        ns, rewards, term, trunc, info = step_batch_jax(s, actions)
        return ns, (rewards, term, trunc)

    @jax.jit
    def run(s):
        return jax.lax.scan(body, s, xs=None, length=steps)

    # warmup
    s2, _ = run(states)
    jax.block_until_ready(s2.time)
    t0 = time.perf_counter()
    s3, outs = run(states)
    jax.block_until_ready(outs[0])
    elapsed = time.perf_counter() - t0
    transitions = num_envs * steps
    return {
        "name": "A_env_transition",
        "num_envs": num_envs,
        "steps": steps,
        "transitions": transitions,
        "elapsed_s": elapsed,
        "tps": transitions / max(elapsed, 1e-9),
        "gpu": _gpu_sample(),
        "peak_vram_mib": _vram_used_mib(),
        "host_rss_bytes": _rss_bytes(),
    }


def bench_env_obs_legal(num_envs: int, steps: int, seed: int = 0) -> dict:
    key = jax.random.PRNGKey(seed)
    pool = build_competition_reset_pool(key, max(num_envs * 4, 64), min_grid=21, max_grid=21)
    states = jax.tree_util.tree_map(lambda x: x[:num_envs], pool)
    mem0 = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_envs), empty_memory())
    mem1 = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_envs), empty_memory())
    actions = _pass_actions(num_envs)

    def body(carry, _):
        s, m0, m1 = carry
        sp0, gv0, m0 = observe_batch_p0(s, m0)
        sp1, gv1, m1 = observe_batch_p1(s, m1)
        mask0 = legal_mask_batch_p0(s)
        mask1 = legal_mask_batch_p1(s)
        ns, rewards, term, trunc, info = step_batch_jax(s, actions)
        return (ns, m0, m1), (sp0, mask0, rewards)

    @jax.jit
    def run(c):
        return jax.lax.scan(body, c, xs=None, length=steps)

    warm, _ = run((states, mem0, mem1))
    jax.block_until_ready(warm[0].time)
    t0 = time.perf_counter()
    out_c, outs = run((states, mem0, mem1))
    jax.block_until_ready(outs[0])
    elapsed = time.perf_counter() - t0
    transitions = num_envs * steps
    return {
        "name": "B_env_obs_legal",
        "num_envs": num_envs,
        "steps": steps,
        "transitions": transitions,
        "elapsed_s": elapsed,
        "tps": transitions / max(elapsed, 1e-9),
        "gpu": _gpu_sample(),
        "peak_vram_mib": _vram_used_mib(),
        "host_rss_bytes": _rss_bytes(),
    }


def bench_rollout(num_envs: int, rollout_len: int, seed: int = 0, reset_pool_size: int = 2048) -> dict:
    key = jax.random.PRNGKey(seed)
    params = init_params(key)
    k_pool, _ = jax.random.split(key)
    pool = build_competition_reset_pool(k_pool, reset_pool_size)
    jax.block_until_ready(jax.tree_util.tree_leaves(pool)[0])
    # warmup
    _ = collect_selfplay_batch(
        params, num_envs=num_envs, rollout_len=rollout_len, seed=seed, reset_pool_size=reset_pool_size, pool=pool
    )
    t0 = time.perf_counter()
    batch = collect_selfplay_batch(
        params, num_envs=num_envs, rollout_len=rollout_len, seed=seed + 1, reset_pool_size=reset_pool_size, pool=pool
    )
    jax.block_until_ready(batch["rewards"])
    elapsed = time.perf_counter() - t0
    transitions = int(batch["rewards"].size)
    return {
        "name": "C_complete_rollout",
        "num_envs": num_envs,
        "rollout_len": rollout_len,
        "reset_pool_size": reset_pool_size,
        "transitions": transitions,
        "elapsed_s": elapsed,
        "tps": transitions / max(elapsed, 1e-9),
        "gpu": _gpu_sample(),
        "peak_vram_mib": _vram_used_mib(),
        "host_rss_bytes": _rss_bytes(),
        "pool_reused": True,
    }


def bench_ppo_only(num_envs: int, rollout_len: int, seed: int = 0, reset_pool_size: int = 2048) -> dict:
    key = jax.random.PRNGKey(seed)
    params = init_params(key)
    batch = collect_selfplay_batch(
        params, num_envs=num_envs, rollout_len=rollout_len, seed=seed, reset_pool_size=reset_pool_size
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
    # GAE separate timing
    t_g0 = time.perf_counter()
    adv, ret = gae_advantages_batch_jit(batch["rewards"], vals, batch["dones"])
    jax.block_until_ready(adv)
    gae_s = time.perf_counter() - t_g0
    flat["advantages"] = adv.reshape(T * N)
    flat["returns"] = ret.reshape(T * N)
    opt = make_optimizer(3e-4)
    opt_state = opt.init(params)
    # warmup PPO
    params, opt_state, _ = ppo_update(params, opt_state, opt, flat)
    jax.block_until_ready(jax.tree_util.tree_leaves(params)[0])
    t0 = time.perf_counter()
    params, opt_state, metrics = ppo_update(params, opt_state, opt, flat)
    jax.block_until_ready(jax.tree_util.tree_leaves(params)[0])
    elapsed = time.perf_counter() - t0
    samples = T * N
    return {
        "name": "D_ppo_update",
        "num_envs": num_envs,
        "rollout_len": rollout_len,
        "samples": samples,
        "gae_elapsed_s": gae_s,
        "gae_samples_per_s": samples / max(gae_s, 1e-9),
        "ppo_elapsed_s": elapsed,
        "ppo_samples_per_s": samples / max(elapsed, 1e-9),
        "joint_gae_ppo_samples_per_s": samples / max(gae_s + elapsed, 1e-9),
        "gpu": _gpu_sample(),
        "peak_vram_mib": _vram_used_mib(),
        "host_rss_bytes": _rss_bytes(),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }


def bench_valid_learning(
    num_envs: int,
    rollout_len: int,
    *,
    updates: int = 3,
    reset_pool_size: int = 4096,
    out_dir: Path,
) -> dict:
    rep = _train_loop(
        out_dir,
        kind="v42_bench_e",
        max_transitions=num_envs * rollout_len * updates,
        max_updates=updates,
        max_seconds=900.0,
        num_envs=num_envs,
        rollout_len=rollout_len,
        seed=0,
        reset_pool_size=reset_pool_size,
    )
    return {
        "name": "E_valid_learning",
        "num_envs": num_envs,
        "rollout_len": rollout_len,
        "reset_pool_size": reset_pool_size,
        "valid_learning_tps": rep["valid_learning_tps"],
        "transitions": rep["transitions"],
        "updates": rep["updates"],
        "elapsed_s": rep["elapsed_s"],
        "compilation_s": rep["compilation_s"],
        "peak_vram_mib": rep["peak_vram_mib"],
        "host_rss_bytes": rep["host_rss_bytes"],
        "gpu": _gpu_sample(),
        "status": rep["status"],
    }


def run_ceiling_ladder(num_envs: int = 32, steps: int = 64, rollout_len: int = 16) -> list[dict]:
    rows = []
    rows.append(bench_env_transition(num_envs, steps))
    rows.append(bench_env_obs_legal(num_envs, steps))
    # stage 4 = complete rollout (obs+legal+transformer)
    rows.append(bench_rollout(num_envs, rollout_len, reset_pool_size=2048))
    # stage 5 = rollout + GAE (reuse D gae part)
    d = bench_ppo_only(num_envs, rollout_len, reset_pool_size=2048)
    rows.append(
        {
            "name": "ceiling_5_rollout_plus_gae",
            "tps_note": "gae_samples_per_s is advantage path only; rollout separate in C",
            "gae_samples_per_s": d["gae_samples_per_s"],
            "ppo_samples_per_s": d["ppo_samples_per_s"],
        }
    )
    return rows


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    out_manifest = repo / "experiments/manifests/competition_native_jax_v4_2_matched_benchmarks.json"
    out_report = repo / "experiments/reports/competition_native_jax_v4_2_matched_benchmarks.md"
    bench_dir = repo / "experiments/competition_native_jax/v4_2_matched_benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)

    num_envs, rollout_len, pool = 32, 16, 4096
    results: dict[str, Any] = {
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_V4_2_MATCHED_BENCHMARKS",
        "config": {
            "num_envs": num_envs,
            "rollout_len": rollout_len,
            "reset_pool_size": pool,
            "observation_dtype": "float32",
            "transformer_dtype": "float32",
            "legal_support": "dense_bool_3970",
            "ppo_epochs": 1,
            "advantage_fraction": 1.0,
            "ppo_semantics": "FULL_BATCH_ONE_OPTAX_UPDATE",
        },
        "lineage": lineage_hashes(),
        "device": detect_jax_device(),
        "benchmarks": {},
        "ceiling_ladder": [],
        "first_collapse_stage": None,
    }

    print("=== A ===", flush=True)
    results["benchmarks"]["A"] = bench_env_transition(num_envs, rollout_len * 4)
    print("=== B ===", flush=True)
    results["benchmarks"]["B"] = bench_env_obs_legal(num_envs, rollout_len * 4)
    print("=== C ===", flush=True)
    results["benchmarks"]["C"] = bench_rollout(num_envs, rollout_len, reset_pool_size=pool)
    print("=== D ===", flush=True)
    results["benchmarks"]["D"] = bench_ppo_only(num_envs, rollout_len, reset_pool_size=pool)
    print("=== E ===", flush=True)
    results["benchmarks"]["E"] = bench_valid_learning(
        num_envs, rollout_len, updates=3, reset_pool_size=pool, out_dir=bench_dir / "E"
    )
    print("=== ceiling ===", flush=True)
    results["ceiling_ladder"] = run_ceiling_ladder(num_envs, steps=rollout_len * 4, rollout_len=rollout_len)

    # Collapse heuristic: successive TPS drops >4x from previous ceiling stage A/B/C
    tps_seq = [
        results["benchmarks"]["A"]["tps"],
        results["benchmarks"]["B"]["tps"],
        results["benchmarks"]["C"]["tps"],
        results["benchmarks"]["D"]["ppo_samples_per_s"],
        results["benchmarks"]["E"]["valid_learning_tps"],
    ]
    names = ["A_transition", "B_obs_legal", "C_rollout", "D_ppo", "E_valid_learning"]
    collapse = None
    for i in range(1, len(tps_seq)):
        if tps_seq[i - 1] > 0 and tps_seq[i] < tps_seq[i - 1] / 4.0:
            collapse = names[i]
            break
    results["first_collapse_stage"] = collapse
    results["tps_sequence"] = dict(zip(names, tps_seq))

    out_manifest.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    md = [
        "# V4.2 matched benchmarks",
        "",
        f"Config: num_envs={num_envs}, rollout_len={rollout_len}, reset_pool_size={pool}",
        "",
        f"First collapse stage: `{collapse}`",
        "",
        "| Bench | Rate | Notes |",
        "|---|---:|---|",
        f"| A env transition TPS | {results['benchmarks']['A']['tps']:.2f} | |",
        f"| B env+obs+legal TPS | {results['benchmarks']['B']['tps']:.2f} | |",
        f"| C complete rollout TPS | {results['benchmarks']['C']['tps']:.2f} | |",
        f"| D PPO samples/s | {results['benchmarks']['D']['ppo_samples_per_s']:.2f} | GAE {results['benchmarks']['D']['gae_samples_per_s']:.2f} |",
        f"| E valid-learning TPS | {results['benchmarks']['E']['valid_learning_tps']:.2f} | |",
        "",
        f"Lineage: `{json.dumps(results['lineage'])}`",
        "",
    ]
    out_report.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"collapse": collapse, "E_tps": results["benchmarks"]["E"]["valid_learning_tps"]}, indent=2))


if __name__ == "__main__":
    main()
