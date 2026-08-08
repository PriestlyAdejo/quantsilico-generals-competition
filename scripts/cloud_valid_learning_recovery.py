#!/usr/bin/env python3
"""Gated A100 pilot and long PPO for CLOUD_VALID_LEARNING_RECOVERY_V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.88")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")

import jax
import jax.numpy as jnp
import numpy as np

from generals_bot.competition_native_jax.competition_env_jax import build_competition_reset_pool
from generals_bot.competition_native_jax.inference_jax import masked_log_softmax
from generals_bot.competition_native_jax.transformer_jax import forward_batch, init_params
from train.competition_native_jax.bc_warmstart_jax import sha256_file
from train.competition_native_jax.checkpoint_recovery import (
    persisted_carry,
    restore_carry,
    save_checkpoint,
)
from train.competition_native_jax.ema_jax import ema_update
from train.competition_native_jax.gae_jax import gae_advantages_batch_jit
from train.competition_native_jax.opponents_jax import build_static_schedule
from train.competition_native_jax.ppo_jax import make_optimizer, ppo_update_with_anchor_jit
from train.competition_native_jax.rollout_curriculum_jax import (
    CurriculumCarry,
    collect_curriculum_batch,
    initialise_curriculum_carry,
)
from train.competition_native_jax.train_jax import load_tree

PROGRAMME = "CLOUD_VALID_LEARNING_RECOVERY_V1"
EXPECTED_PARENT_LEARNER = "2b10b1e326ba4f3b6532441b6a9f11fbb696e9d90684c81d6105f893df12ece2"
GAMMA = 1.0
LAMBDA = 0.9
LR = 3e-5
RESET_POOL_SIZE = 4096
MILESTONES = (250_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000, 25_000_000)
STAGES = (
    (0.40, 0.40, 0.20, 0.00),
    (0.25, 0.35, 0.30, 0.10),
    (0.10, 0.25, 0.40, 0.25),
    (0.00, 0.15, 0.45, 0.30, 0.10),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def hash_files(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in sorted(paths, key=str):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def materialized_tree_hash(tree: Any) -> str:
    """Hash array content, dtype and shape for an immutable device tree."""
    digest = hashlib.sha256()
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        value = np.ascontiguousarray(np.asarray(leaf))
        digest.update(str(path).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


def runtime_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    official = root / "third_party" / "generals-bots" / "generals" / "agents"
    return {
        "learner_hash": hash_files(
            (
                root / "src/generals_bot/competition_native_jax/transformer_jax.py",
                root / "train/competition_native_jax/ppo_jax.py",
                root / "train/competition_native_jax/gae_jax.py",
            )
        ),
        "environment_hash": hash_files(
            (root / "src/generals_bot/competition_native_jax/competition_env_jax.py",)
        ),
        "rollout_hash": hash_files(
            (root / "train/competition_native_jax/rollout_curriculum_jax.py",)
        ),
        "reward_hash": hash_files(
            (root / "train/competition_native_jax/rollout_curriculum_jax.py",)
        ),
        "opponent_hash": hash_files(
            (
                root / "train/competition_native_jax/opponents_jax.py",
                official / "random_agent.py",
                official / "expander_agent.py",
                official / "hunter_agent.py",
            )
        ),
        "curriculum_hash": hashlib.sha256(
            json.dumps(STAGES, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "runtime_hash": hash_files(
            (
                Path(__file__).resolve(),
                root / "train/competition_native_jax/rollout_curriculum_jax.py",
                root / "train/competition_native_jax/opponents_jax.py",
                root / "train/competition_native_jax/checkpoint_recovery.py",
            )
        ),
    }


def gpu_snapshot() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
        name, util, used, total, power, temperature = [part.strip() for part in output.split(",")]
        return {
            "name": name,
            "util_pct": float(util),
            "vram_used_mib": float(used),
            "vram_total_mib": float(total),
            "power_w": float(power),
            "temperature_c": float(temperature),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def verify_device() -> dict[str, Any]:
    devices = jax.devices()
    kinds = [getattr(device, "device_kind", str(device)) for device in devices]
    if len(devices) != 1 or jax.default_backend() != "gpu" or "A100" not in kinds[0]:
        raise RuntimeError(
            f"expected exactly one A100 GPU device; backend={jax.default_backend()} kinds={kinds}"
        )
    return {
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "device_count": 1,
        "device_kind": kinds[0],
    }


def load_anchor_dataset(path: Path) -> tuple[dict[str, jax.Array], str]:
    manifest = json.loads(path.with_name("dataset_manifest.json").read_text(encoding="utf-8"))
    actual_sha = sha256_file(path)
    if actual_sha != manifest.get("dataset_sha256"):
        raise RuntimeError("supervised anchor dataset SHA mismatch")
    raw = np.load(path, allow_pickle=False)
    split = np.asarray(raw["split"]).astype(str)
    train_indices = np.flatnonzero(split == "train")
    if not len(train_indices):
        raise RuntimeError("supervised anchor training split is empty")
    anchor = {
        "spatial": jax.device_put(jnp.asarray(raw["spatial"], dtype=jnp.float32)),
        "global": jax.device_put(jnp.asarray(raw["global_vec"], dtype=jnp.float32)),
        "mask": jax.device_put(jnp.asarray(raw["legal_mask"], dtype=bool)),
        "actions": jax.device_put(jnp.asarray(raw["teacher_action"], dtype=jnp.int32)),
        "train_indices": jax.device_put(jnp.asarray(train_indices, dtype=jnp.int32)),
    }
    return anchor, actual_sha


def anchor_batch(anchor: dict[str, jax.Array], seed: int) -> dict[str, jax.Array]:
    positions = jax.random.randint(
        jax.random.PRNGKey(seed),
        (256,),
        0,
        anchor["train_indices"].shape[0],
    )
    selected = anchor["train_indices"][positions]
    return {key: value[selected] for key, value in anchor.items() if key != "train_indices"}


def flatten_batch(batch: dict[str, Any]) -> tuple[dict[str, jax.Array], jax.Array, jax.Array]:
    if not bool(jnp.all(batch["learner_controlled_mask"])):
        raise RuntimeError("LEARNER_CONTROLLED_MASK_GATE failed")
    steps, envs = batch["rewards"].shape
    values = jnp.concatenate([batch["values"], batch["bootstrap_values"][None, :]], axis=0)
    advantages, returns = gae_advantages_batch_jit(
        batch["rewards"], values, batch["dones"], gamma=GAMMA, lam=LAMBDA
    )
    advantage_std = jnp.std(advantages)
    normalized = (advantages - jnp.mean(advantages)) / jnp.maximum(advantage_std, 1e-8)
    flat = {
        "spatial": batch["spatial"].reshape((steps * envs,) + batch["spatial"].shape[2:]),
        "global": batch["global"].reshape((steps * envs,) + batch["global"].shape[2:]),
        "mask": batch["mask"].reshape(steps * envs, -1),
        "actions": batch["actions"].reshape(steps * envs),
        "old_logp": batch["old_logp"].reshape(steps * envs),
        "advantages": normalized.reshape(steps * envs),
        "returns": returns.reshape(steps * envs),
    }
    return flat, advantages, returns


@jax.jit
def post_update_diagnostics(params: dict, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
    output = forward_batch(params, batch["spatial"], batch["global"])
    logp = jax.vmap(masked_log_softmax)(output["flat_logits"], batch["mask"])
    selected = jnp.take_along_axis(logp, batch["actions"][:, None], axis=1)[:, 0]
    ratio = jnp.exp(selected - batch["old_logp"])
    legal = jnp.take_along_axis(batch["mask"], batch["actions"][:, None], axis=1)[:, 0]
    return {
        "approx_kl": jnp.mean(batch["old_logp"] - selected),
        "clip_fraction": jnp.mean(jnp.abs(ratio - 1.0) > 0.2),
        "ratio_mean": jnp.mean(ratio),
        "legality_faults": jnp.sum(~legal),
        "support_faults": jnp.sum(jnp.sum(batch["mask"], axis=1) == 0),
        "nonfinite_output_faults": jnp.sum(~jnp.isfinite(selected)),
    }


def one_update(
    params: dict,
    ema: dict,
    opt_state: Any,
    optimizer: Any,
    carry: CurriculumCarry,
    schedule: tuple[int, ...],
    anchor: dict[str, jax.Array],
    *,
    anchor_seed: int,
    rollout_len: int,
    shaping_lambda: float,
) -> tuple[dict, dict, Any, CurriculumCarry, dict[str, Any]]:
    before = params
    rollout_started = time.perf_counter()
    batch, carry = collect_curriculum_batch(
        params,
        opponent_schedule=schedule,
        rollout_len=rollout_len,
        carry=carry,
        gamma=GAMMA,
        shaping_lambda=shaping_lambda,
    )
    rollout_s = time.perf_counter() - rollout_started
    flat, advantages, returns = flatten_batch(batch)
    learner_started = time.perf_counter()
    params, opt_state, loss_metrics = ppo_update_with_anchor_jit(
        params,
        opt_state,
        optimizer,
        flat,
        anchor_batch(anchor, anchor_seed),
        accumulate_minibatches=8,
    )
    ema = ema_update(ema, params)
    diagnostics = post_update_diagnostics(params, flat)
    jax.block_until_ready(jax.tree_util.tree_leaves(params)[0])
    learner_s = time.perf_counter() - learner_started
    update_norm = jnp.sqrt(
        sum(
            jnp.sum(jnp.square(new - old))
            for new, old in zip(
                jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(before), strict=True
            )
        )
    )
    host = {key: np.asarray(value) for key, value in batch.items() if hasattr(value, "shape")}
    completed = int(np.count_nonzero(host["dones"]))
    wins = int(np.count_nonzero(host["learner_won"]))
    losses = int(np.count_nonzero(host["learner_lost"]))
    truncated = int(np.count_nonzero(host["truncated"]))
    samples = int(host["rewards"].size)
    returns_np = np.asarray(returns)
    values_np = np.asarray(batch["values"])
    return_variance = float(np.var(returns_np))
    explained_variance = (
        1.0 - float(np.var(returns_np - values_np)) / return_variance
        if return_variance > 1e-12
        else 0.0
    )
    metrics = {
        "samples": samples,
        "rollout_s": rollout_s,
        "learner_s": learner_s,
        "valid_learning_tps": samples / max(rollout_s + learner_s, 1e-9),
        "raw_tps": samples / max(rollout_s + learner_s, 1e-9),
        "completed_episodes": completed,
        "wins": wins,
        "losses": losses,
        "draws": truncated,
        "nonzero_terminal_rewards": int(np.count_nonzero(host["terminal_rewards"])),
        "nonzero_rewards": int(np.count_nonzero(host["rewards"])),
        "reward_nonzero_fraction": float(np.count_nonzero(host["rewards"]) / samples),
        "terminal_reward_mean": float(np.mean(host["terminal_rewards"])),
        "shaped_reward_mean": float(np.mean(host["shaped_rewards"])),
        "return_std": float(jnp.std(returns)),
        "advantage_std": float(jnp.std(advantages)),
        "explained_variance": explained_variance,
        "pass_fraction": float(np.mean(host["learner_pass"])),
        "shaping_lambda": shaping_lambda,
        "learning_rate": LR,
        "parameter_update_norm": float(update_norm),
        **{key: float(value) for key, value in loss_metrics.items()},
        **{key: float(value) for key, value in diagnostics.items()},
    }
    return params, ema, opt_state, carry, metrics


def compile_without_mutating(
    params: dict,
    optimizer: Any,
    opt_state: Any,
    carry: CurriculumCarry,
    schedule: tuple[int, ...],
    anchor: dict[str, jax.Array],
    rollout_len: int,
) -> None:
    one_update(
        params,
        params,
        opt_state,
        optimizer,
        carry,
        schedule,
        anchor,
        anchor_seed=0,
        rollout_len=rollout_len,
        shaping_lambda=0.0,
    )


def pathology_reasons(metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    finite_fields = (
        "loss",
        "pg",
        "vloss",
        "entropy",
        "approx_kl",
        "return_std",
        "advantage_std",
        "explained_variance",
        "parameter_update_norm",
        "PPO_ACTOR_GRAD_NORM",
        "ANCHOR_RAW_GRAD_NORM",
        "ANCHOR_SCALE",
        "ANCHOR_EFFECTIVE_GRAD_NORM",
        "ANCHOR_TO_PPO_RATIO",
    )
    for field in finite_fields:
        if not np.isfinite(float(metrics.get(field, np.nan))):
            reasons.append(f"nonfinite:{field}")
    if abs(float(metrics.get("approx_kl", 0.0))) > 0.2:
        reasons.append("pathological_kl")
    if float(metrics.get("clip_fraction", 0.0)) > 0.8:
        reasons.append("pathological_clipping")
    for field in ("legality_faults", "support_faults", "nonfinite_output_faults"):
        if int(metrics.get(field, 0)) != 0:
            reasons.append(field)
    if float(metrics.get("NO_TASK_POLICY_GRADIENT", 1.0)) != 0.0:
        reasons.append("NO_TASK_POLICY_GRADIENT")
    return reasons


def run_pilot_once(
    *,
    params: dict,
    runtime: Path,
    shaping_lambda: float,
    pool: Any,
    pool_seed: int,
    pool_materialized_hash: str,
    anchor: dict[str, jax.Array],
    anchor_sha256: str,
) -> tuple[dict[str, Any], Path | None]:
    optimizer = make_optimizer(LR)
    opt_state = optimizer.init(params)
    ema = params
    geometries = ((256, 32, 300_000),)
    rows: list[dict[str, Any]] = []
    total_completed = total_nonzero = total_reward_nonzero = total_samples = 0
    total_wins = total_losses = 0
    final_checkpoint: Path | None = None
    for geometry_index, (num_envs, rollout_len, target) in enumerate(geometries):
        schedule = build_static_schedule(num_envs, STAGES[0])
        carry = initialise_curriculum_carry(
            params,
            num_envs=num_envs,
            seed=101 + geometry_index,
            reset_pool_size=RESET_POOL_SIZE,
            pool=pool,
            frozen_opponent_params=params,
        )
        compile_without_mutating(
            params, optimizer, opt_state, carry, schedule, anchor, rollout_len
        )
        geometry_started = time.perf_counter()
        transitions = 0
        while transitions < target:
            params, ema, opt_state, carry, metrics = one_update(
                params,
                ema,
                opt_state,
                optimizer,
                carry,
                schedule,
                anchor,
                anchor_seed=10_000 + len(rows),
                rollout_len=rollout_len,
                shaping_lambda=shaping_lambda,
            )
            transitions += metrics["samples"]
            total_samples += metrics["samples"]
            total_completed += metrics["completed_episodes"]
            total_nonzero += metrics["nonzero_terminal_rewards"]
            total_reward_nonzero += metrics["nonzero_rewards"]
            total_wins += metrics["wins"]
            total_losses += metrics["losses"]
            reasons = pathology_reasons(metrics)
            rows.append(
                {
                    "geometry": f"{num_envs}x{rollout_len}",
                    "geometry_transitions": transitions,
                    "programme_transitions": total_samples,
                    "shaping_lambda": shaping_lambda,
                    "metrics": metrics,
                    "pathology_reasons": reasons,
                    "ts": utc_now(),
                }
            )
            atomic_json(runtime / "metrics" / "pilot_latest.json", rows[-1])
            append_jsonl(runtime / "metrics" / "pilot.jsonl", rows[-1])
            if reasons:
                break
        elapsed = time.perf_counter() - geometry_started
        rows[-1]["geometry_elapsed_s"] = elapsed
        if rows[-1]["pathology_reasons"]:
            break

    last = rows[-1]["metrics"]
    valid = bool(
        not any(row["pathology_reasons"] for row in rows)
        and total_completed >= 64
        and total_nonzero >= 32
        and total_wins > 0
        and total_losses > 0
        and float(last["return_std"]) > 1e-4
        and float(last["advantage_std"]) > 1e-4
        and float(last["parameter_update_norm"]) > 1e-8
        and float(last["PPO_ACTOR_GRAD_NORM"]) >= 1e-8
        and float(last["NO_TASK_POLICY_GRADIENT"]) == 0.0
        and total_reward_nonzero > 0
    )
    report = {
        "schema_version": 1,
        "kind": "CLOUD_VALID_LEARNING_MICRO_PILOT",
        "status": "CLOUD_VALID_LEARNING_MICRO_PILOT_PASS" if valid else "FAIL",
        "VALID_LEARNING_GATE_PASS": valid,
        "shaping_lambda": shaping_lambda,
        "total_valid_transitions": total_samples,
        "completed_episodes": total_completed,
        "nonzero_terminal_rewards": total_nonzero,
        "terminal_wins": total_wins,
        "terminal_losses": total_losses,
        "reward_nonzero_fraction": total_reward_nonzero / max(total_samples, 1),
        "anchor_dataset_sha256": anchor_sha256,
        "rows": rows,
        "hashes": runtime_hashes(),
        "device": verify_device(),
        "gpu": gpu_snapshot(),
        "written_at": utc_now(),
    }
    if valid:
        meta = {
            "programme": PROGRAMME,
            "update": len(rows),
            "transitions": total_samples,
            "programme_transitions": total_samples,
            "num_envs": 512,
            "rollout_len": 32,
            "curriculum_stage": 0,
            "shaping_lambda": shaping_lambda,
            "gamma": GAMMA,
            "lambda": LAMBDA,
            "reset_pool_seed": pool_seed,
            "reset_pool_size": RESET_POOL_SIZE,
            "pool_identity": hashlib.sha256(
                f"{pool_seed}:{RESET_POOL_SIZE}:18:21".encode()
            ).hexdigest(),
            "pool_materialized_hash": pool_materialized_hash,
            "anchor_dataset_sha256": anchor_sha256,
            **runtime_hashes(),
        }
        final_checkpoint = save_checkpoint(
            runtime / "training" / "checkpoints",
            tag="pilot",
            params=params,
            ema=ema,
            opt_state=opt_state,
            carry=carry,
            meta=meta,
        )
        report["checkpoint"] = str(final_checkpoint)
    atomic_json(runtime / "pilot_report.json", report)
    return report, final_checkpoint


def run_pilot(args: argparse.Namespace) -> int:
    device = verify_device()
    params = load_tree(args.calibrated, init_params(jax.random.PRNGKey(0)))
    params = jax.device_put(params)
    anchor, anchor_sha256 = load_anchor_dataset(args.anchor_dataset)
    pool_seed = 131
    pool_started = time.perf_counter()
    pool = build_competition_reset_pool(jax.random.PRNGKey(pool_seed), RESET_POOL_SIZE)
    jax.block_until_ready(jax.tree_util.tree_leaves(pool)[0])
    pool_build_s = time.perf_counter() - pool_started
    pool_materialized_hash = materialized_tree_hash(pool)
    report, checkpoint = run_pilot_once(
        params=params,
        runtime=args.runtime,
        shaping_lambda=0.0,
        pool=pool,
        pool_seed=pool_seed,
        pool_materialized_hash=pool_materialized_hash,
        anchor=anchor,
        anchor_sha256=anchor_sha256,
    )
    pathology = any(row["pathology_reasons"] for row in report["rows"])
    if (
        report["status"] != "CLOUD_VALID_LEARNING_MICRO_PILOT_PASS"
        and args.allow_shaping_repair
        and not pathology
    ):
        report, checkpoint = run_pilot_once(
            params=params,
            runtime=args.runtime / "shaping_repair",
            shaping_lambda=0.05,
            pool=pool,
            pool_seed=pool_seed,
            pool_materialized_hash=pool_materialized_hash,
            anchor=anchor,
            anchor_sha256=anchor_sha256,
        )
    report["pool_build_s"] = pool_build_s
    report["device"] = device
    report["checkpoint"] = str(checkpoint) if checkpoint else None
    atomic_json(args.runtime / "cloud_valid_learning_micro_pilot.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if checkpoint else 2


def load_checkpoint_for_long(
    path: Path, pool: Any
) -> tuple[dict, dict, Any, CurriculumCarry, dict]:
    template = init_params(jax.random.PRNGKey(0))
    params = load_tree(path / "raw.npz", template)
    ema = load_tree(path / "ema.npz", template)
    optimizer = make_optimizer(LR)
    opt_state = load_tree(path / "opt_state.npz", optimizer.init(template))
    frozen = load_tree(path / "frozen_opponent.npz", template)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    carry_template = persisted_carry(
        initialise_curriculum_carry(
            template,
            num_envs=int(meta["num_envs"]),
            seed=0,
            reset_pool_size=RESET_POOL_SIZE,
            pool=pool,
        )
    )
    saved = load_tree(path / "rollout_carry.npz", carry_template)
    carry = restore_carry(saved, params=params, pool=pool, frozen_opponent_params=frozen)
    return params, ema, opt_state, carry, meta


def run_long(args: argparse.Namespace) -> int:
    device = verify_device()
    stop_at = datetime.fromisoformat(args.stop_at.replace("Z", "+00:00"))
    pilot_meta = json.loads((args.parent / "meta.json").read_text(encoding="utf-8"))
    pool_seed = int(pilot_meta["reset_pool_seed"])
    pool = build_competition_reset_pool(jax.random.PRNGKey(pool_seed), RESET_POOL_SIZE)
    jax.block_until_ready(jax.tree_util.tree_leaves(pool)[0])
    actual_pool_hash = materialized_tree_hash(pool)
    expected_pool_hash = pilot_meta.get("pool_materialized_hash")
    if not expected_pool_hash or actual_pool_hash != expected_pool_hash:
        raise RuntimeError(
            "materialized reset-pool hash mismatch: "
            f"expected={expected_pool_hash} actual={actual_pool_hash}"
        )
    params, ema, opt_state, carry, meta0 = load_checkpoint_for_long(args.parent, pool)
    anchor, anchor_sha256 = load_anchor_dataset(args.anchor_dataset)
    if meta0.get("anchor_dataset_sha256") != anchor_sha256:
        raise RuntimeError("long-run anchor dataset differs from pilot")
    identities = runtime_hashes()
    optimizer = make_optimizer(LR)
    stage = int(meta0.get("curriculum_stage", 0))
    schedule = build_static_schedule(args.num_envs, STAGES[stage])
    if len(carry.learner_seat) != args.num_envs:
        carry = initialise_curriculum_carry(
            params,
            num_envs=args.num_envs,
            seed=211,
            reset_pool_size=RESET_POOL_SIZE,
            pool=pool,
            frozen_opponent_params=params,
        )
    compile_without_mutating(
        params, optimizer, opt_state, carry, schedule, anchor, args.rollout_len
    )
    started = time.perf_counter()
    programme_start = int(meta0["programme_transitions"])
    transitions = int(meta0["transitions"])
    programme_transitions = int(meta0["programme_transitions"])
    update = int(meta0["update"])
    milestones = deque(item for item in MILESTONES if item > programme_transitions)
    outcomes: deque[int] = deque(maxlen=256)
    qualifying_windows = 0
    regressing_windows = 0
    zero_signal_samples = zero_signal_completions = zero_signal_terminals = 0
    exit_reason = "TRAINING_BUDGET_STOP"

    while datetime.now(UTC) < stop_at:
        stop_request = args.runtime / "STOP_REQUEST"
        if stop_request.exists():
            exit_reason = "STOP_REQUEST"
            break
        update += 1
        params, ema, opt_state, carry, metrics = one_update(
            params,
            ema,
            opt_state,
            optimizer,
            carry,
            schedule,
            anchor,
            anchor_seed=20_000 + update,
            rollout_len=args.rollout_len,
            shaping_lambda=float(meta0.get("shaping_lambda", 0.0)),
        )
        transitions += metrics["samples"]
        programme_transitions += metrics["samples"]
        outcomes.extend([1] * metrics["wins"])
        outcomes.extend([-1] * metrics["losses"])
        outcomes.extend([0] * metrics["draws"])
        zero_signal_samples += metrics["samples"]
        zero_signal_completions += metrics["completed_episodes"]
        zero_signal_terminals += metrics["nonzero_terminal_rewards"]
        reasons = pathology_reasons(metrics)
        if reasons:
            exit_reason = "PPO_PATHOLOGY_ABORT:" + ",".join(reasons)
            break
        if (
            zero_signal_samples >= 500_000
            and zero_signal_completions >= 64
            and zero_signal_terminals == 0
        ):
            exit_reason = "NO_TASK_LEARNING_SIGNAL"
            break
        if zero_signal_terminals > 0:
            zero_signal_samples = zero_signal_completions = zero_signal_terminals = 0

        if len(outcomes) >= 256:
            window = list(outcomes)[-128:]
            completion = len(window)
            wins = sum(item == 1 for item in window)
            losses = sum(item == -1 for item in window)
            draws = completion - wins - losses
            # High win-rate is positive evidence, not a reason to hold the easy
            # stage forever.  A persistently loss-dominated harder stage is a
            # reason to step back.  Require two consecutive windows either way.
            advance = (
                stage < len(STAGES) - 1
                and
                completion >= 128
                and draws / completion <= 0.70
                and wins >= max(losses, 1)
            )
            regress = (
                stage > 0
                and completion >= 128
                and losses / completion >= 0.80
            )
            qualifying_windows = qualifying_windows + 1 if advance else 0
            regressing_windows = regressing_windows + 1 if regress else 0
            if qualifying_windows >= 2:
                stage += 1
                schedule = build_static_schedule(args.num_envs, STAGES[stage])
                carry = initialise_curriculum_carry(
                    params,
                    num_envs=args.num_envs,
                    seed=211 + stage,
                    reset_pool_size=RESET_POOL_SIZE,
                    pool=pool,
                    frozen_opponent_params=params,
                )
                qualifying_windows = 0
                regressing_windows = 0
                outcomes.clear()
            elif regressing_windows >= 2:
                stage -= 1
                schedule = build_static_schedule(args.num_envs, STAGES[stage])
                carry = initialise_curriculum_carry(
                    params,
                    num_envs=args.num_envs,
                    seed=307 + stage,
                    reset_pool_size=RESET_POOL_SIZE,
                    pool=pool,
                    frozen_opponent_params=params,
                )
                qualifying_windows = 0
                regressing_windows = 0
                outcomes.clear()

        elapsed = time.perf_counter() - started
        row = {
            "schema_version": 1,
            "kind": "CLOUD_VALID_LEARNING_LATEST",
            "status": "RUNNING",
            "pid": os.getpid(),
            "update": update,
            "transitions": transitions,
            "programme_transitions": programme_transitions,
            "session_transitions": programme_transitions - programme_start,
            "sustained_valid_learning_tps": (programme_transitions - programme_start)
            / max(elapsed, 1e-9),
            "curriculum_stage": stage,
            "opponent_counts": [schedule.count(kind) for kind in range(len(STAGES[stage]))],
            "hashes": identities,
            "metrics": metrics,
            "gpu": gpu_snapshot() if update % 10 == 0 else None,
            "training_stop_at": args.stop_at,
            "ts": utc_now(),
        }
        if update % 5 == 0:
            atomic_json(args.runtime / "metrics" / "cloud_training_latest.json", row)
            append_jsonl(args.runtime / "metrics" / "cloud_training.jsonl", row)
            print(
                f"HB update={update} valid={programme_transitions} "
                f"tps={row['sustained_valid_learning_tps']:.1f} "
                f"completed={metrics['completed_episodes']} "
                f"terminal={metrics['nonzero_terminal_rewards']}",
                flush=True,
            )
        while milestones and programme_transitions >= milestones[0]:
            milestone = milestones.popleft()
            checkpoint = save_checkpoint(
                args.runtime / "training" / "checkpoints",
                tag=f"plus_{milestone}",
                params=params,
                ema=ema,
                opt_state=opt_state,
                carry=carry,
                meta={
                    **meta0,
                    "programme": PROGRAMME,
                    "update": update,
                    "transitions": transitions,
                    "programme_transitions": programme_transitions,
                    "milestone": milestone,
                    "curriculum_stage": stage,
                    "num_envs": args.num_envs,
                    "rollout_len": args.rollout_len,
                    **identities,
                },
            )
            print(f"MILESTONE_COMPLETE={checkpoint}", flush=True)

    final_checkpoint = save_checkpoint(
        args.runtime / "training" / "checkpoints",
        tag="final",
        params=params,
        ema=ema,
        opt_state=opt_state,
        carry=carry,
        meta={
            **meta0,
            "programme": PROGRAMME,
            "update": update,
            "transitions": transitions,
            "programme_transitions": programme_transitions,
            "curriculum_stage": stage,
            "num_envs": args.num_envs,
            "rollout_len": args.rollout_len,
            "exit_reason": exit_reason,
            **identities,
        },
    )
    state = {
        "schema_version": 1,
        "kind": "CLOUD_VALID_LEARNING_PROGRAMME_STATE",
        "status": "COMPLETE",
        "exit_reason": exit_reason,
        "final_checkpoint": str(final_checkpoint),
        "update": update,
        "transitions": transitions,
        "programme_transitions": programme_transitions,
        "curriculum_stage": stage,
        "device": device,
        "gpu": gpu_snapshot(),
        "written_at": utc_now(),
    }
    atomic_json(args.runtime / "programme_state.json", state)
    atomic_json(args.runtime / "metrics" / "cloud_training_latest.json", state)
    print(json.dumps(state, indent=2), flush=True)
    return 0 if exit_reason in ("TRAINING_BUDGET_STOP", "STOP_REQUEST") else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    pilot = commands.add_parser("pilot")
    pilot.add_argument("--calibrated", type=Path, required=True)
    pilot.add_argument("--anchor-dataset", type=Path, required=True)
    pilot.add_argument("--allow-shaping-repair", action="store_true")
    train = commands.add_parser("train")
    train.add_argument("--parent", type=Path, required=True)
    train.add_argument("--anchor-dataset", type=Path, required=True)
    train.add_argument("--num-envs", type=int, default=512)
    train.add_argument("--rollout-len", type=int, default=32)
    train.add_argument("--stop-at", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.runtime.mkdir(parents=True, exist_ok=True)
    if args.command == "pilot":
        return run_pilot(args)
    return run_long(args)


if __name__ == "__main__":
    raise SystemExit(main())
