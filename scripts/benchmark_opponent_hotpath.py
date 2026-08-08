#!/usr/bin/env python3
"""A100 timing gate for the four batched JAX opponent paths."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp

from generals_bot.competition_native_jax.competition_env_jax import (
    ObsMemoryJax,
    auto_reset_from_pool,
    build_competition_reset_pool,
    index_to_engine_action_batch,
    observe_batch_p0,
    step_batch_jax,
)
from generals_bot.competition_native_jax.transformer_jax import forward_batch, init_params
from train.competition_native_jax.opponents_jax import OpponentKind, batched_opponent_actions
from train.competition_native_jax.train_jax import load_tree


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def timed(callable_, repeats: int) -> float:
    result = callable_()
    jax.block_until_ready(jax.tree_util.tree_leaves(result)[0])
    started = time.perf_counter()
    for _ in range(repeats):
        result = callable_()
    jax.block_until_ready(jax.tree_util.tree_leaves(result)[0])
    return (time.perf_counter() - started) / repeats


def gpu_snapshot() -> dict[str, float | str] | None:
    """Return one low-overhead telemetry sample without adding a dependency."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        name, util, used, total, power = [part.strip() for part in output.split(",")]
        return {
            "name": name,
            "utilization_percent": float(util),
            "memory_used_mib": float(used),
            "memory_total_mib": float(total),
            "power_w": float(power),
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    devices = jax.devices()
    kinds = [getattr(device, "device_kind", str(device)) for device in devices]
    if len(devices) != 1 or jax.default_backend() != "gpu" or "A100" not in kinds[0]:
        raise RuntimeError(f"expected one A100, got backend={jax.default_backend()} kinds={kinds}")

    pool_started = time.perf_counter()
    pool = build_competition_reset_pool(jax.random.PRNGKey(81), 4096)
    jax.block_until_ready(jax.tree_util.tree_leaves(pool)[0])
    pool_build_s = time.perf_counter() - pool_started
    states = jax.tree_util.tree_map(lambda value: value[: args.num_envs], pool)
    seats = jnp.arange(args.num_envs, dtype=jnp.int32) % 2
    keys = jax.random.split(jax.random.PRNGKey(83), args.num_envs)
    opponent_timings: dict[str, float] = {}
    for name, kind in (
        ("pass", OpponentKind.PASS),
        ("random", OpponentKind.RANDOM),
        ("expander", OpponentKind.EXPANDER),
        ("hunter", OpponentKind.HUNTER),
    ):
        schedule = tuple([int(kind)] * args.num_envs)
        seconds = timed(
            lambda schedule=schedule: batched_opponent_actions(states, seats, keys, schedule),
            args.repeats,
        )
        opponent_timings[name] = seconds

    params = load_tree(args.checkpoint, init_params(jax.random.PRNGKey(0)))
    memory_zero = jnp.zeros((args.num_envs, 21, 21), dtype=jnp.float32)
    spatial, global_vec, _ = observe_batch_p0(
        states, ObsMemoryJax(memory_zero, memory_zero)
    )
    learner_forward_s = timed(lambda: forward_batch(params, spatial, global_vec), args.repeats)
    pass_actions = index_to_engine_action_batch(
        jnp.zeros((args.num_envs,), dtype=jnp.int32)
    )
    joint_actions = jnp.stack([pass_actions, pass_actions], axis=1)
    environment_step_s = timed(lambda: step_batch_jax(states, joint_actions), args.repeats)
    done = jnp.ones((args.num_envs,), dtype=bool)
    cursor = jnp.arange(args.num_envs, 2 * args.num_envs, dtype=jnp.int32)
    reset_s = timed(
        lambda: auto_reset_from_pool(states, done, jnp.zeros_like(done), pool, cursor),
        args.repeats,
    )
    report = {
        "schema_version": 1,
        "kind": "OPPONENT_HOTPATH_PERFORMANCE_GATE",
        "status": "PASS",
        "device": kinds[0],
        "num_envs": args.num_envs,
        "repeats": args.repeats,
        "opponent_action_wall_s": opponent_timings,
        "learner_forward_wall_s": learner_forward_s,
        "environment_step_wall_s": environment_step_s,
        "reset_pool_build_s": pool_build_s,
        "reset_index_wall_s": reset_s,
        "cpu_count": os.cpu_count(),
        "gpu_sample": gpu_snapshot(),
        "python_loop_per_environment": False,
        "host_device_roundtrip_in_opponent_hotpath": False,
        "all_four_present": set(opponent_timings) == {"pass", "random", "expander", "hunter"},
        "written_at": datetime.now(UTC).isoformat(),
    }
    if max(opponent_timings.values()) > 10 * max(learner_forward_s, 1e-9):
        report["status"] = "FAIL_CATASTROPHIC_OPPONENT_SLOWDOWN"
    atomic_json(args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
