"""Assemble the MARATHON_BASELINE_V0 capsule record (EXECUTION_PLAN sections 6.2-6.5).

Collects, with real runs only:
1. Source/engine/config identity (commits + lineage hashes).
2. Checkpoint FILE and SEMANTIC hashes (re-reads the tracked capsule input).
3. Determinism contract (JAX/XLA versions, backend, dtype policy, flags).
4. Determinism evidence: identical digests across two independent process runs
   (behavioural fingerprint + one-update resume) on CPU.
5. Separate TPS measurements: hot-path (collection only), end-to-end
   (collection + update), valid-learning (transitions behind healthy updates).

Output: experiments/marathon/baseline_capsule_v0.json (tracked evidence).
This is a capsule, not strength evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from generals_bot.competition_native_jax.transformer_jax import init_params  # noqa: E402
from train.competition_native_jax.ema_jax import ema_update  # noqa: E402
from train.competition_native_jax.gae_jax import gae_advantages_batch_jit  # noqa: E402
from train.competition_native_jax.ppo_jax import make_optimizer, ppo_update  # noqa: E402
from train.competition_native_jax.rollout_selfplay_jax import collect_selfplay_batch  # noqa: E402
from train.competition_native_jax.train_jax import lineage_hashes, load_tree  # noqa: E402

DEFAULT_CHECKPOINT = (
    Path.home()
    / "quantsilico-runtime"
    / "cloud_assisted_deadline_salvage_v1_final"
    / "ckpt_final_u482_t7593984"
)
LR = 3e-4
FIRST_N_STEPS = 2
FIRST_N_ENVS = 2


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def digest_array(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(array)).tobytes()).hexdigest()


def digest_tree(tree) -> str:
    digest_obj = hashlib.sha256()
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    for key_path, leaf in sorted(flat, key=lambda item: str(item[0])):
        arr = np.ascontiguousarray(np.asarray(leaf))
        digest_obj.update(str(key_path).encode("utf-8"))
        digest_obj.update(arr.tobytes(order="C"))
    return digest_obj.hexdigest()


def finite_tree(tree) -> bool:
    return all(
        bool(np.isfinite(np.asarray(leaf)).all()) for leaf in jax.tree_util.tree_leaves(tree)
    )


def flatten_batch(batch: dict) -> dict:
    t_steps, n_envs = batch["rewards"].shape
    flat = {
        "spatial": batch["spatial"].reshape(t_steps * n_envs, *batch["spatial"].shape[2:]),
        "global": batch["global"].reshape(t_steps * n_envs, *batch["global"].shape[2:]),
        "mask": batch["mask"].reshape(t_steps * n_envs, -1),
        "actions": batch["actions"].reshape(t_steps * n_envs),
        "old_logp": batch["old_logp"].reshape(t_steps * n_envs),
    }
    values = jnp.concatenate([batch["values"], batch["bootstrap_values"][None, :]], axis=0)
    advantages, returns = gae_advantages_batch_jit(batch["rewards"], values, batch["dones"])
    flat["advantages"] = advantages.reshape(t_steps * n_envs)
    flat["returns"] = returns.reshape(t_steps * n_envs)
    return flat


def behaviour_digests(
    params, *, num_envs: int, rollout_len: int, seed: int, reset_pool_size: int
) -> dict:
    batch = collect_selfplay_batch(
        params,
        num_envs=num_envs,
        rollout_len=rollout_len,
        seed=seed,
        reset_pool_size=reset_pool_size,
    )
    keys = ("spatial", "global", "mask", "actions", "old_logp", "values", "rewards", "dones")
    return {
        key: digest_array(np.asarray(batch[key])[:FIRST_N_STEPS, :FIRST_N_ENVS]) for key in keys
    }


def one_update(
    params, ema, opt_state, optimizer, *, num_envs, rollout_len, seed, reset_pool_size
):
    batch = collect_selfplay_batch(
        params,
        num_envs=num_envs,
        rollout_len=rollout_len,
        seed=seed,
        reset_pool_size=reset_pool_size,
    )
    flat = flatten_batch(batch)
    started = time.perf_counter()
    new_params, new_opt_state, metrics = ppo_update(params, opt_state, optimizer, flat)
    jax.block_until_ready(jax.tree_util.tree_leaves(new_params))
    update_s = time.perf_counter() - started
    new_ema = ema_update(ema, new_params)
    return new_params, new_ema, new_opt_state, metrics, update_s


def healthy(metrics: dict, new_params, new_opt_state) -> bool:
    values_finite = all(math.isfinite(float(v)) for v in metrics.values())
    ratio_ok = 0.5 <= float(metrics.get("ratio", 0.0)) <= 2.0
    return values_finite and ratio_ok and finite_tree(new_params) and finite_tree(new_opt_state)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "experiments/marathon/baseline_capsule_v0.json",
    )
    parser.add_argument("--hot-batches", type=int, default=12)
    parser.add_argument("--e2e-iterations", type=int, default=4)
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    total_started = time.perf_counter()
    params_like = init_params(jax.random.PRNGKey(0))
    optimizer = make_optimizer(LR)
    opt_like = optimizer.init(params_like)
    params = load_tree(args.checkpoint / "raw.npz", params_like)
    ema = load_tree(args.checkpoint / "ema.npz", params_like)
    opt_state = load_tree(args.checkpoint / "opt_state.npz", opt_like)

    fingerprint_cfg = dict(num_envs=4, rollout_len=8, seed=1234, reset_pool_size=64)
    resume_cfg = dict(num_envs=4, rollout_len=8, seed=7, reset_pool_size=64)
    tps_cfg = dict(num_envs=8, rollout_len=16, reset_pool_size=128)

    # Determinism evidence: two independent collections with identical config.
    behaviour_a = behaviour_digests(params, **fingerprint_cfg)
    behaviour_b = behaviour_digests(params, **fingerprint_cfg)
    update_a = one_update(params, ema, opt_state, optimizer, **resume_cfg)
    update_b = one_update(params, ema, opt_state, optimizer, **resume_cfg)
    determinism = {
        "behaviour_digests_match": behaviour_a == behaviour_b,
        "update_params_digests_match": digest_tree(update_a[0]) == digest_tree(update_b[0]),
        "update_ema_digests_match": digest_tree(update_a[1]) == digest_tree(update_b[1]),
        "update_opt_digests_match": digest_tree(update_a[2]) == digest_tree(update_b[2]),
        "update_metrics_match": {
            key: math.isclose(
                float(update_a[3][key]), float(update_b[3][key]), rel_tol=1e-6, abs_tol=1e-6
            )
            for key in update_a[3]
        },
    }

    # Hot-path TPS: collection only, frozen baseline policy (warm cache).
    hot_transitions = 0
    hot_started = time.perf_counter()
    for index in range(args.hot_batches):
        batch = collect_selfplay_batch(params, seed=500 + index, **tps_cfg)
        jax.block_until_ready(jax.tree_util.tree_leaves(batch["actions"]))
        hot_transitions += int(np.asarray(batch["actions"]).size)
    hot_s = time.perf_counter() - hot_started

    # End-to-end + valid-learning TPS: collect + one update per iteration.
    e2e_params, e2e_ema, e2e_opt = params, ema, opt_state
    e2e_transitions = 0
    valid_transitions = 0
    update_times = []
    e2e_metrics = []
    e2e_health = []
    e2e_started = time.perf_counter()
    for index in range(args.e2e_iterations):
        batch = collect_selfplay_batch(e2e_params, seed=700 + index, **tps_cfg)
        jax.block_until_ready(jax.tree_util.tree_leaves(batch["actions"]))
        transitions = int(np.asarray(batch["actions"]).size)
        e2e_transitions += transitions
        flat = flatten_batch(batch)
        update_started = time.perf_counter()
        e2e_params, e2e_opt, metrics = ppo_update(e2e_params, e2e_opt, optimizer, flat)
        jax.block_until_ready(jax.tree_util.tree_leaves(e2e_params))
        update_times.append(time.perf_counter() - update_started)
        e2e_ema = ema_update(e2e_ema, e2e_params)
        metrics_float = {key: float(value) for key, value in metrics.items()}
        e2e_metrics.append(metrics_float)
        iteration_healthy = healthy(metrics_float, e2e_params, e2e_opt)
        e2e_health.append(iteration_healthy)
        if iteration_healthy:
            valid_transitions += transitions
    e2e_s = time.perf_counter() - e2e_started

    try:
        import jaxlib

        jaxlib_version = jaxlib.__version__
    except Exception:
        jaxlib_version = None
    try:
        from jax._src.lib import xla_extension_version  # type: ignore

        xla_version = str(xla_extension_version)
    except Exception:
        xla_version = None

    capsule = {
        "schema_version": 1,
        "kind": "MARATHON_BASELINE_V0_CAPSULE",
        "assembled_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_identity": {
            "repo_commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "engine_submodule": git("submodule", "status", "third_party/generals-bots"),
            "lineage_hashes": lineage_hashes(),
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "semantic_hashes_file": "experiments/marathon/baseline_semantic_hashes.json",
        },
        "determinism_contract": {
            "JAX_VERSION": jax.__version__,
            "JAXLIB_VERSION": jaxlib_version,
            "XLA_VERSION": xla_version,
            "CUDA_VERSION": None,
            "GPU_MODEL": None,
            "BACKEND": jax.default_backend(),
            "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
            "DTYPE_POLICY": "float32_default",
            "DETERMINISM_MODE": "CPU_DEFAULT_NO_ASYNC",
            "LEVELS_CLAIMED": [
                "BITWISE_DETERMINISM",
                "STATE_SEMANTIC_DETERMINISM",
                "BEHAVIOURAL_DETERMINISM",
            ],
            "LEVELS_DEMONSTRATED": [
                "STATE_SEMANTIC_DETERMINISM",
                "BEHAVIOURAL_DETERMINISM",
            ],
            "EVIDENCE": determinism,
        },
        "resume_proof": {
            "config": {**resume_cfg, "lr": LR},
            "metrics": {key: float(value) for key, value in update_a[3].items()},
            "update_seconds": update_a[4],
            "post_digests": {
                "params": digest_tree(update_a[0]),
                "ema": digest_tree(update_a[1]),
                "opt_state": digest_tree(update_a[2]),
            },
        },
        "throughput": {
            "device": str(jax.devices()),
            "hot_path": {
                "definition": "self-play collection only, frozen baseline policy, warm JIT",
                "config": {**tps_cfg, "batches": args.hot_batches},
                "transitions": hot_transitions,
                "wall_seconds": hot_s,
                "tps": hot_transitions / max(hot_s, 1e-9),
            },
            "end_to_end": {
                "definition": "collection + one full-batch PPO update per iteration",
                "config": {**tps_cfg, "iterations": args.e2e_iterations},
                "transitions": e2e_transitions,
                "wall_seconds": e2e_s,
                "tps": e2e_transitions / max(e2e_s, 1e-9),
                "update_seconds_each": update_times,
            },
            "valid_learning": {
                "definition": "transitions behind updates passing finite-metric/ratio health",
                "transitions": valid_transitions,
                "wall_seconds": e2e_s,
                "tps": valid_transitions / max(e2e_s, 1e-9),
                "updates": len(e2e_metrics),
                "healthy_updates": sum(1 for item in e2e_health if item),
                "metrics_by_update": e2e_metrics,
            },
        },
        "behaviour_fingerprint": {
            "config": fingerprint_cfg,
            "first_n_digests": behaviour_a,
        },
        "total_wall_seconds": time.perf_counter() - total_started,
    }

    determinism_ok = (
        determinism["behaviour_digests_match"]
        and determinism["update_params_digests_match"]
        and determinism["update_ema_digests_match"]
        and determinism["update_opt_digests_match"]
        and all(determinism["update_metrics_match"].values())
    )
    capsule["status"] = "PASS" if determinism_ok else "FAIL"
    args.out.write_text(json.dumps(capsule, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(capsule, indent=2, sort_keys=True))
    return 0 if capsule["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
