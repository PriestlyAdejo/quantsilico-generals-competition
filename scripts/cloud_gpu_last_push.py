#!/usr/bin/env python3
"""CLOUD_GPU_LAST_PUSH_V1 geometry gate and detached long-PPO runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")

import jax
import jax.numpy as jnp
import numpy as np

from generals_bot.competition_native_jax.competition_env_jax import build_competition_reset_pool
from generals_bot.competition_native_jax.inference_jax import masked_log_softmax
from generals_bot.competition_native_jax.obs_memory import N_GLOBAL, N_SPATIAL
from generals_bot.competition_native_jax.transformer_jax import forward, forward_batch, init_params
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

PROGRAMME = "CLOUD_GPU_LAST_PUSH_V1"
EXPECTED_PARENT_LEARNER = "2b10b1e326ba4f3b6532441b6a9f11fbb696e9d90684c81d6105f893df12ece2"
DEFAULT_PARENT = Path("/workspace/quantsilico-runtime/cloud_gpu_last_push_v1/parent_u1524")
DEFAULT_RUNTIME = Path("/workspace/quantsilico-runtime/cloud_gpu_last_push_v1")
RESET_POOL_SIZE = 4096
LR = 3e-4
GEOMETRIES = ((256, 32), (128, 64), (512, 32))
MILESTONES = (5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000, 200_000_000, 400_000_000, 700_000_000)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    tmp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_snapshot() -> dict[str, Any]:
    try:
        text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
        name, util, used, total, power, temp = [part.strip() for part in text.split(",")]
        return {
            "name": name,
            "util_pct": float(util),
            "vram_used_mib": float(used),
            "vram_total_mib": float(total),
            "power_w": float(power),
            "temperature_c": float(temp),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def verify_device() -> dict[str, Any]:
    device = detect_jax_device()
    devices = jax.devices()
    kinds = [getattr(item, "device_kind", str(item)) for item in devices]
    if len(devices) != 1 or jax.default_backend() != "gpu" or "A100" not in kinds[0]:
        raise RuntimeError(f"expected exactly one A100 GPU JAX device, got backend={jax.default_backend()} kinds={kinds}")
    return {**device, "device_count": len(devices), "device_kinds": kinds}


def initialise_templates():
    key = jax.random.PRNGKey(0)
    params_like = jax.device_put(init_params(key))
    _ = forward(params_like, jnp.zeros((N_SPATIAL, 21, 21)), jnp.zeros((N_GLOBAL,)))
    optimizer = make_optimizer(LR)
    opt_like = optimizer.init(params_like)
    return params_like, optimizer, opt_like


def validate_parent(meta: dict[str, Any]) -> None:
    lineage = meta.get("lineage") or {}
    learner = lineage.get("learner_implementation_hash")
    if learner != EXPECTED_PARENT_LEARNER:
        raise RuntimeError(f"immutable parent learner mismatch: {learner}")
    if int(meta.get("update", -1)) != 1524 or int(meta.get("transitions", -1)) != 1_560_576:
        raise RuntimeError(f"immutable parent counters mismatch: {meta.get('update')} {meta.get('transitions')}")


def flatten_batch(batch: dict[str, Any]) -> tuple[dict[str, jax.Array], jax.Array, jax.Array]:
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
    return flat, advantages, returns


@jax.jit
def post_update_diagnostics(params: dict, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
    outputs = forward_batch(params, batch["spatial"], batch["global"])
    logp_all = jax.vmap(masked_log_softmax)(outputs["flat_logits"], batch["mask"])
    new_logp = jnp.take_along_axis(logp_all, batch["actions"][:, None], axis=1)[:, 0]
    ratio = jnp.exp(new_logp - batch["old_logp"])
    centers = jnp.linspace(-1.0, 1.0, outputs["value_logits"].shape[-1])
    value_pred = jnp.sum(jax.nn.softmax(outputs["value_logits"], axis=-1) * centers, axis=-1)
    returns = batch["returns"]
    return_var = jnp.var(returns)
    explained = jnp.where(return_var > 1e-12, 1.0 - jnp.var(returns - value_pred) / return_var, 0.0)
    chosen_legal = jnp.take_along_axis(batch["mask"], batch["actions"][:, None], axis=1)[:, 0]
    finite = jnp.logical_and(jnp.isfinite(new_logp), jnp.isfinite(value_pred))
    return {
        "approx_kl": jnp.mean(batch["old_logp"] - new_logp),
        "clip_fraction": jnp.mean(jnp.abs(ratio - 1.0) > 0.2),
        "explained_variance": explained,
        "ratio_post_mean": jnp.mean(ratio),
        "legality_faults": jnp.sum(~chosen_legal),
        "support_faults": jnp.sum(jnp.sum(batch["mask"], axis=1) == 0),
        "nonfinite_output_faults": jnp.sum(~finite),
    }


def one_update(
    params,
    ema,
    opt_state,
    optimizer,
    *,
    num_envs: int,
    rollout_len: int,
    pool,
    seed: int,
    diagnostics: bool,
):
    batch = collect_selfplay_batch(
        params,
        num_envs=num_envs,
        rollout_len=rollout_len,
        seed=seed,
        reset_pool_size=RESET_POOL_SIZE,
        pool=pool,
    )
    flat, _advantages, returns = flatten_batch(batch)
    before = params
    params, opt_state, metrics = ppo_update(params, opt_state, optimizer, flat)
    ema = ema_update(ema, params)
    jax.block_until_ready(jax.tree_util.tree_leaves(params)[0])
    grad_like_update = jax.tree_util.tree_map(lambda new, old: new - old, params, before)
    update_norm = float(optax_global_norm(grad_like_update))
    host_rewards = np.asarray(batch["rewards"])
    host_dones = np.asarray(batch["dones"])
    host_returns = np.asarray(returns)
    result = {key: float(value) for key, value in metrics.items()}
    result.update(
        {
            "reward_mean": float(host_rewards.mean()),
            "reward_nonzero_fraction": float(np.count_nonzero(host_rewards) / host_rewards.size),
            "completion_rate": float(host_dones.mean()),
            "return_mean": float(host_returns.mean()),
            "return_std": float(host_returns.std()),
            "parameter_update_norm": update_norm,
        }
    )
    if diagnostics:
        result.update({key: float(value) for key, value in post_update_diagnostics(params, flat).items()})
    return params, ema, opt_state, result, int(host_rewards.size)


def optax_global_norm(tree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(item)) for item in leaves))


def checkpoint_atomic(
    root: Path,
    tag: str,
    *,
    params,
    ema,
    opt_state,
    meta: dict[str, Any],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"ckpt_{tag}"
    tmp = root / f"ckpt_{tag}.tmp-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save_tree(tmp / "raw.npz", params)
    save_tree(tmp / "ema.npz", ema)
    save_tree(tmp / "opt_state.npz", opt_state)
    atomic_json(tmp / "meta.json", meta)
    _ = load_training_checkpoint(tmp, params_like=params, opt_state_like=opt_state)
    files = {}
    for name in ("raw.npz", "ema.npz", "opt_state.npz", "meta.json"):
        path = tmp / name
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    atomic_json(tmp / "sha256_manifest.json", {"schema_version": 1, "files": files})
    for path in tmp.iterdir():
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    if final.exists():
        shutil.rmtree(final)
    tmp.replace(final)
    atomic_json(
        final / "COMPLETE",
        {"ok": True, "tag": tag, "update": meta["update"], "transitions": meta["transitions"], "written_at": utc_now()},
    )
    return final


def healthy(metrics: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for index, row in enumerate(metrics):
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                reasons.append(f"nonfinite:{index}:{key}")
        if row.get("legality_faults", 0) or row.get("support_faults", 0) or row.get("nonfinite_output_faults", 0):
            reasons.append(f"faults:{index}")
        if abs(float(row.get("approx_kl", 0.0))) > 0.25:
            reasons.append(f"pathological_kl:{index}")
        if float(row.get("clip_fraction", 0.0)) > 0.8:
            reasons.append(f"pathological_clip:{index}")
    return not reasons, reasons


def run_geometry(args: argparse.Namespace) -> int:
    device = verify_device()
    params_like, optimizer, opt_like = initialise_templates()
    base = load_training_checkpoint(args.parent, params_like=params_like, opt_state_like=opt_like)
    validate_parent(base["meta"])
    pool = build_competition_reset_pool(jax.random.PRNGKey(7), RESET_POOL_SIZE)
    jax.block_until_ready(jax.tree_util.tree_leaves(pool)[0])
    gate_start = time.perf_counter()
    rows = []
    geometry_root = args.runtime / "geometry_gate"
    for candidate_index, (num_envs, rollout_len) in enumerate(GEOMETRIES):
        if time.perf_counter() - gate_start >= args.wall_seconds:
            raise RuntimeError("GEOMETRY_GATE_WALL_CLOCK_CAP reached before all candidates completed")
        loaded = load_training_checkpoint(args.parent, params_like=params_like, opt_state_like=opt_like)
        params = jax.device_put(loaded["params"])
        ema = jax.device_put(loaded["ema"])
        opt_state = loaded["opt_state"]
        seed_base = 10_000 + candidate_index * 10_000
        params, ema, opt_state, _warm, _ = one_update(
            params,
            ema,
            opt_state,
            optimizer,
            num_envs=num_envs,
            rollout_len=rollout_len,
            pool=pool,
            seed=seed_base,
            diagnostics=True,
        )
        target_updates = math.ceil(args.transitions / (num_envs * rollout_len))
        metrics: list[dict[str, Any]] = []
        transitions = 0
        started = time.perf_counter()
        for update_index in range(target_updates):
            if time.perf_counter() - gate_start >= args.wall_seconds:
                raise RuntimeError("GEOMETRY_GATE_WALL_CLOCK_CAP reached during candidate")
            params, ema, opt_state, row, count = one_update(
                params,
                ema,
                opt_state,
                optimizer,
                num_envs=num_envs,
                rollout_len=rollout_len,
                pool=pool,
                seed=seed_base + update_index + 1,
                diagnostics=update_index == 0 or update_index == target_updates - 1 or update_index % 5 == 4,
            )
            transitions += count
            row.update({"update_index": update_index + 1, "transitions": transitions})
            metrics.append(row)
        elapsed = time.perf_counter() - started
        is_healthy, reasons = healthy(metrics)
        tag = f"e{num_envs}_r{rollout_len}"
        meta = {
            **loaded["meta"],
            "update": int(loaded["meta"]["update"]) + target_updates,
            "transitions": int(loaded["meta"]["transitions"]) + transitions,
            "programme_start_transitions": int(loaded["meta"]["transitions"]),
            "next_seed": seed_base + target_updates + 1,
            "num_envs": num_envs,
            "rollout_len": rollout_len,
            "reset_pool_size": RESET_POOL_SIZE,
            "parent_checkpoint": str(args.parent),
            "parent_checkpoint_learner": EXPECTED_PARENT_LEARNER,
            "runtime_learner": lineage_hashes()["learner_implementation_hash"],
            "checkpoint_compatible_continuation": True,
            "exact_frozen_source_continuation": False,
            "geometry_pilot": True,
        }
        checkpoint = checkpoint_atomic(geometry_root / tag, "final", params=params, ema=ema, opt_state=opt_state, meta=meta)
        rows.append(
            {
                "tag": tag,
                "num_envs": num_envs,
                "rollout_len": rollout_len,
                "status": "HEALTHY" if is_healthy else "REJECTED_PATHOLOGY",
                "reasons": reasons,
                "updates": target_updates,
                "transitions": transitions,
                "elapsed_s": elapsed,
                "valid_learning_tps": transitions / max(elapsed, 1e-9),
                "gpu": gpu_snapshot(),
                "metrics_first": metrics[0],
                "metrics_last": metrics[-1],
                "checkpoint": str(checkpoint),
            }
        )
        atomic_json(
            args.runtime / "cloud_training_geometry_gate.json",
            {"schema_version": 1, "kind": "CLOUD_TRAINING_GEOMETRY_GATE", "status": "PARTIAL", "rows": rows},
        )
    valid = [row for row in rows if row["status"] == "HEALTHY"]
    if not valid:
        raise RuntimeError("all geometry candidates rejected")
    winner = max(valid, key=lambda row: float(row["valid_learning_tps"]))
    report = {
        "schema_version": 1,
        "kind": "CLOUD_TRAINING_GEOMETRY_GATE",
        "programme": PROGRAMME,
        "status": "COMPLETE",
        "selection_policy": "fixed_500k_health_gate_then_fastest_stable_when_no_material_short-horizon_learning_difference",
        "wall_clock_cap_s": args.wall_seconds,
        "elapsed_s": time.perf_counter() - gate_start,
        "parent": str(args.parent),
        "device": device,
        "rows": rows,
        "winner": winner,
        "written_at": utc_now(),
    }
    atomic_json(args.runtime / "cloud_training_geometry_gate.json", report)
    print(json.dumps({"GEOMETRY_GATE": "COMPLETE", "winner": winner}, indent=2), flush=True)
    return 0


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_train(args: argparse.Namespace) -> int:
    device = verify_device()
    params_like, optimizer, opt_like = initialise_templates()
    loaded = load_training_checkpoint(args.parent, params_like=params_like, opt_state_like=opt_like)
    params = jax.device_put(loaded["params"])
    ema = jax.device_put(loaded["ema"])
    opt_state = loaded["opt_state"]
    meta0 = loaded["meta"]
    programme_start = int(meta0.get("programme_start_transitions", 1_560_576))
    transitions = int(meta0["transitions"])
    updates = int(meta0["update"])
    seed = int(meta0.get("next_seed", updates + 1))
    pool = build_competition_reset_pool(jax.random.PRNGKey(int(meta0.get("reset_pool_seed") or 7)), RESET_POOL_SIZE)
    jax.block_until_ready(jax.tree_util.tree_leaves(pool)[0])
    stop_at = datetime.fromisoformat(args.stop_at.replace("Z", "+00:00"))
    stop_request = args.runtime / "training" / "STOP_REQUEST"
    checkpoints = args.runtime / "training" / "checkpoints"
    metrics_root = args.runtime / "training" / "metrics"
    pending = [item for item in MILESTONES if transitions - programme_start < item]
    started = time.perf_counter()
    session_start = transitions
    last_metrics: dict[str, Any] = {}
    exit_reason = "TRAINING_BUDGET_STOP_AT"
    while datetime.now(timezone.utc) < stop_at and not stop_request.exists():
        updates += 1
        params, ema, opt_state, last_metrics, count = one_update(
            params,
            ema,
            opt_state,
            optimizer,
            num_envs=args.num_envs,
            rollout_len=args.rollout_len,
            pool=pool,
            seed=seed,
            diagnostics=updates % 10 == 0,
        )
        seed += 1
        transitions += count
        elapsed = time.perf_counter() - started
        programme_delta = transitions - programme_start
        row = {
            "schema_version": 1,
            "kind": "CLOUD_LONG_PPO_LATEST",
            "programme": PROGRAMME,
            "status": "RUNNING",
            "pid": os.getpid(),
            "update": updates,
            "transitions": transitions,
            "programme_transitions": programme_delta,
            "session_transitions": transitions - session_start,
            "tps": (transitions - session_start) / max(elapsed, 1e-9),
            "elapsed_s": elapsed,
            "num_envs": args.num_envs,
            "rollout_len": args.rollout_len,
            "next_milestone": pending[0] if pending else None,
            "training_budget_stop_at": args.stop_at,
            "last_metrics": last_metrics,
            "gpu": gpu_snapshot() if updates % 10 == 0 else None,
            "ts": utc_now(),
        }
        if updates == int(meta0["update"]) + 1 or updates % 5 == 0:
            atomic_json(metrics_root / "cloud_training_latest.json", row)
            append_jsonl(metrics_root / "cloud_training_metrics.jsonl", row)
            print(
                f"HB update={updates} transitions={transitions} programme_delta={programme_delta} tps={row['tps']:.2f}",
                flush=True,
            )
        while pending and programme_delta >= pending[0]:
            milestone = pending.pop(0)
            meta = {
                **meta0,
                "update": updates,
                "transitions": transitions,
                "programme_start_transitions": programme_start,
                "next_seed": seed,
                "num_envs": args.num_envs,
                "rollout_len": args.rollout_len,
                "reset_pool_size": RESET_POOL_SIZE,
                "parent_checkpoint": str(args.parent),
                "runtime_learner": lineage_hashes()["learner_implementation_hash"],
                "checkpoint_compatible_continuation": True,
                "exact_frozen_source_continuation": False,
                "milestone_delta": milestone,
            }
            saved = checkpoint_atomic(checkpoints, f"plus_{milestone}", params=params, ema=ema, opt_state=opt_state, meta=meta)
            print(f"MILESTONE_COMPLETE delta={milestone} path={saved}", flush=True)
    if stop_request.exists():
        exit_reason = "STOP_REQUEST_FILE"
    final_meta = {
        **meta0,
        "update": updates,
        "transitions": transitions,
        "programme_start_transitions": programme_start,
        "next_seed": seed,
        "num_envs": args.num_envs,
        "rollout_len": args.rollout_len,
        "reset_pool_size": RESET_POOL_SIZE,
        "parent_checkpoint": str(args.parent),
        "runtime_learner": lineage_hashes()["learner_implementation_hash"],
        "checkpoint_compatible_continuation": True,
        "exact_frozen_source_continuation": False,
        "exit_reason": exit_reason,
    }
    final_path = checkpoint_atomic(checkpoints, "final", params=params, ema=ema, opt_state=opt_state, meta=final_meta)
    final = {
        "schema_version": 1,
        "kind": "CLOUD_LONG_PPO_STATE",
        "programme": PROGRAMME,
        "status": "COMPLETE",
        "exit_reason": exit_reason,
        "update": updates,
        "transitions": transitions,
        "programme_transitions": transitions - programme_start,
        "final_checkpoint": str(final_path),
        "training_budget_stop_at": args.stop_at,
        "device": device,
        "last_metrics": last_metrics,
        "written_at": utc_now(),
    }
    atomic_json(args.runtime / "programme_state.json", final)
    atomic_json(metrics_root / "cloud_training_latest.json", final)
    print(json.dumps(final, indent=2), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    sub = parser.add_subparsers(dest="command", required=True)
    gate = sub.add_parser("geometry")
    gate.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    gate.add_argument("--transitions", type=int, default=500_000)
    gate.add_argument("--wall-seconds", type=int, default=2700)
    train = sub.add_parser("train")
    train.add_argument("--parent", type=Path, required=True)
    train.add_argument("--num-envs", type=int, required=True)
    train.add_argument("--rollout-len", type=int, required=True)
    train.add_argument("--stop-at", required=True, help="RFC3339 UTC training-stop ceiling")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.runtime.mkdir(parents=True, exist_ok=True)
    if args.command == "geometry":
        return run_geometry(args)
    return run_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
