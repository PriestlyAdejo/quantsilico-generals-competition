"""Emergency exact-resume PPO from R-E.6 ckpt_final (parent learner 2b10…)."""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LEARNER = "2b10b1e326ba4f3b6532441b6a9f11fbb696e9d90684c81d6105f893df12ece2"
RUNTIME = Path.home() / "quantsilico-runtime" / "emergency_rolling_v1"
# Default parent is R-E.6; cooperative distill handoff may override via env.
PARENT = Path(
    os.environ.get(
        "EMERGENCY_RESUME_PARENT",
        str(ROOT / "experiments/competition_native_jax/v4_3_r_e6/ckpt_final"),
    )
)
OUT = RUNTIME / "training"
CKPT_ROOT = OUT / "checkpoints"
METRICS = OUT / "metrics"
REPO_MIRROR = ROOT / "experiments/competition_native_jax/emergency_rolling_v1"


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    with tmp.open("rb") as f:
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    tmp.replace(path)


def _set_gpu_owner(owner: str) -> None:
    doc = {
        "schema_version": 1,
        "kind": "EMERGENCY_GPU_OWNERSHIP",
        "gpu_owner": owner,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
    }
    _atomic_write_json(RUNTIME / "gpu" / "gpu_owner.json", doc)
    _atomic_write_json(ROOT / "experiments/manifests/emergency_gpu_owner.json", doc)


def _save_ckpt_atomic(
    tag: str,
    *,
    params,
    ema,
    opt_state,
    meta: dict[str, Any],
) -> Path:
    """checkpoint_name.tmp/ → fsync → verify load → rename → COMPLETE."""
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    final = CKPT_ROOT / f"ckpt_{tag}"
    tmp = CKPT_ROOT / f"ckpt_{tag}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save_tree(tmp / "raw.npz", params)
    save_tree(tmp / "ema.npz", ema)
    save_tree(tmp / "opt_state.npz", opt_state)
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    # flush best-effort
    for p in tmp.iterdir():
        with p.open("rb") as f:
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
    # local verify load
    _ = load_training_checkpoint(tmp, params_like=params, opt_state_like=opt_state)
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    (final / "COMPLETE").write_text(
        json.dumps(
            {
                "ok": True,
                "tag": tag,
                "update": meta.get("update"),
                "manifest_sha256_note": "meta.json present; consumers require COMPLETE",
                "written_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    # Lightweight Windows-visible pointer only (avoid copying multi‑GB trees onto /mnt/c).
    mirror_meta = REPO_MIRROR / "checkpoints" / f"ckpt_{tag}"
    mirror_meta.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final / "meta.json", mirror_meta / "meta.json")
    shutil.copy2(final / "COMPLETE", mirror_meta / "COMPLETE")
    (mirror_meta / "RUNTIME_PATH.txt").write_text(str(final) + "\n", encoding="utf-8")
    return final


def _append_metrics(row: dict) -> None:
    METRICS.mkdir(parents=True, exist_ok=True)
    latest = METRICS / "emergency_training_latest.json"
    jsonl = METRICS / "emergency_training_metrics.jsonl"
    chart = METRICS / "emergency_training_chart.json"
    _atomic_write_json(latest, row)
    with jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    # bounded chart ≤1000 pts
    pts: list[dict] = []
    if chart.exists():
        try:
            pts = json.loads(chart.read_text(encoding="utf-8")).get("points", [])
        except Exception:
            pts = []
    pts.append(
        {
            "update": row["updates"],
            "transitions": row["transitions"],
            "tps": row["tps"],
            "entropy": row.get("entropy"),
            "loss": row.get("loss"),
            "ts": row["ts"],
        }
    )
    if len(pts) > 1000:
        # downsample keep last 500 + every Nth of older
        older = pts[:-500]
        step = max(1, len(older) // 500)
        pts = older[::step][-500:] + pts[-500:]
    _atomic_write_json(chart, {"schema_version": 1, "kind": "EMERGENCY_TRAINING_CHART", "points": pts[-1000:]})
    # mirror latest into repo
    mirror_m = REPO_MIRROR / "metrics"
    mirror_m.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, mirror_m / "emergency_training_latest.json")
    shutil.copy2(chart, mirror_m / "emergency_training_chart.json")


def _deadline_hit() -> tuple[bool, str]:
    dpath = ROOT / "experiments/manifests/emergency_deadlines.json"
    if not dpath.exists():
        return False, ""
    d = json.loads(dpath.read_text(encoding="utf-8"))
    # wall clock
    stop_at = datetime.fromisoformat(d["experiments_stop_at"])
    if datetime.now(timezone.utc) >= stop_at:
        return True, "EXPERIMENTS_STOP_AT"
    # monotonic
    mono_deadline = float(d.get("experiments_stop_monotonic_deadline_s", 0))
    if mono_deadline and time.monotonic() >= mono_deadline:
        return True, "EXPERIMENTS_STOP_MONOTONIC"
    return False, ""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    REPO_MIRROR.mkdir(parents=True, exist_ok=True)
    stop_path = OUT / "STOP_REQUEST"
    stop_path_repo = REPO_MIRROR / "STOP_REQUEST"

    lin = lineage_hashes()
    if lin["learner_implementation_hash"] != EXPECTED_LEARNER:
        print("BLOCKED_EMERGENCY_LINEAGE", lin["learner_implementation_hash"])
        return 2

    runtime = json.loads((ROOT / "experiments/manifests/daytime_runtime_selected.json").read_text(encoding="utf-8"))
    freeze = json.loads(
        (ROOT / "experiments/manifests/competition_native_jax_daytime_eval_protocol_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    deadlines = json.loads((ROOT / "experiments/manifests/emergency_deadlines.json").read_text(encoding="utf-8"))
    sel = runtime["selected"]
    num_envs = int(sel["num_envs"])
    rollout_len = int(sel["rollout_len"])
    reset_pool_size = int(sel["reset_pool_size"])
    # Emergency: continue until experiments_stop or +6e6 transitions (plan)
    max_additional_transitions = 6_000_000
    protocol_id = freeze["evaluation_protocol_id"]
    protocol_sha = freeze["evaluation_protocol_sha256"]

    device = detect_jax_device()
    print("EMERGENCY_RESUME start", json.dumps({"device": device, "sel": sel}), flush=True)
    _set_gpu_owner("PPO_TRAINER")

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
        print("BLOCKED_EMERGENCY_LINEAGE parent", parent_learner)
        _set_gpu_owner("NONE")
        return 2

    updates = int(meta0["update"])
    transitions = int(meta0["transitions"])
    start_updates = updates
    start_transitions = transitions
    lr = float(meta0.get("lr", 3e-4))
    if abs(lr - 3e-4) > 1e-12:
        print("WARN lr mismatch", lr)

    # restore RNG / pool from meta
    mr = meta0.get("model_rng") or [0, 0]
    key = jnp.array(mr, dtype=jnp.uint32)
    if key.shape[0] < 2:
        key = jax.random.PRNGKey(int(mr[0]) if mr else 0)
    pool_seed_u32 = int(meta0.get("reset_pool_seed") or 0)
    pool_cursor = int(meta0.get("reset_pool_cursor") or 0)
    key_pool = jax.random.PRNGKey(pool_seed_u32)
    reset_pool = build_competition_reset_pool(key_pool, reset_pool_size)
    jax.block_until_ready(jax.tree_util.tree_leaves(reset_pool)[0])
    pool_refresh_every = 10

    # warm compile with current shapes
    warm = collect_selfplay_batch(
        params,
        num_envs=num_envs,
        rollout_len=rollout_len,
        seed=updates + 1,
        reset_pool_size=reset_pool_size,
        pool=reset_pool,
    )
    jax.block_until_ready(warm["rewards"])

    t0 = time.perf_counter()
    peak_vram = _vram_used_mib() or 0.0
    last_metrics: dict = {}
    status = "COMPLETED"
    exit_reason = "BUDGET_OR_TIME"
    ckpt_paths: list[str] = []
    next_fixed = ((updates // 256) + 1) * 256

    launch = {
        "schema_version": 1,
        "kind": "EMERGENCY_EXACT_RESUME_LAUNCH",
        "parent": str(PARENT),
        "parent_update": start_updates,
        "parent_transitions": start_transitions,
        "selected": sel,
        "max_additional_transitions": max_additional_transitions,
        "experiments_stop_at": deadlines.get("experiments_stop_at"),
        "learner": EXPECTED_LEARNER,
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
    }
    _atomic_write_json(ROOT / "experiments/manifests/emergency_exact_resume_launch.json", launch)
    _atomic_write_json(RUNTIME / "programme" / "exact_resume_launch.json", launch)

    try:
        while True:
            if stop_path.exists() or stop_path_repo.exists():
                status = "ABORTED_STOP_REQUEST"
                exit_reason = "STOP_REQUEST_FILE"
                break
            hit, why = _deadline_hit()
            if hit:
                status = "COMPLETED"
                exit_reason = why
                break
            if (transitions - start_transitions) >= max_additional_transitions:
                status = "COMPLETED"
                exit_reason = "MAX_ADDITIONAL_TRANSITIONS"
                break

            if updates > 0 and updates % pool_refresh_every == 0:
                key, key_pool = jax.random.split(key)
                reset_pool = build_competition_reset_pool(key_pool, reset_pool_size)
                jax.block_until_ready(jax.tree_util.tree_leaves(reset_pool)[0])
                pool_seed_u32 = int(jnp.asarray(key_pool).reshape(-1)[0])
                pool_cursor = 0

            batch = collect_selfplay_batch(
                params,
                num_envs=num_envs,
                rollout_len=rollout_len,
                seed=updates + 1,
                reset_pool_size=reset_pool_size,
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
            params, opt_state, last_metrics = ppo_update(params, opt_state, optimizer, flat)
            ema = ema_update(ema, params)
            transitions += T * N
            updates += 1
            elapsed = time.perf_counter() - t0
            tps = (transitions - start_transitions) / max(elapsed, 1e-6)
            v = _vram_used_mib()
            if v is not None:
                peak_vram = max(peak_vram, v)

            row = {
                "schema_version": 1,
                "kind": "EMERGENCY_TRAINING_LATEST",
                "updates": updates,
                "transitions": transitions,
                "session_transitions": transitions - start_transitions,
                "tps": tps,
                "elapsed_s": elapsed,
                "entropy": float(last_metrics.get("entropy", 0.0)) if last_metrics else None,
                "loss": float(last_metrics.get("loss", 0.0)) if last_metrics else None,
                "pg": float(last_metrics.get("pg", 0.0)) if last_metrics else None,
                "ratio": float(last_metrics.get("ratio", 0.0)) if last_metrics else None,
                "vloss": float(last_metrics.get("vloss", 0.0)) if last_metrics else None,
                "peak_vram_mib": peak_vram,
                "gpu_owner": "PPO_TRAINER",
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            if updates == start_updates + 1 or updates % 10 == 0:
                _append_metrics(row)
                print(
                    f"HB updates={updates} transitions={transitions} session_tps={tps:.2f}",
                    flush=True,
                )

            save_now = False
            tag = None
            if updates == start_updates + 1:
                save_now, tag = True, f"u{updates}"
            elif updates >= next_fixed:
                save_now, tag = True, f"u{updates}"
                next_fixed += 256
            if save_now and tag:
                meta = {
                    "update": updates,
                    "transitions": transitions,
                    "lr": 3e-4,
                    "model_rng": [int(x) for x in list(jnp.asarray(key).reshape(-1)[:2])],
                    "env_rng_seed_base": 0,
                    "reset_pool_seed": pool_seed_u32,
                    "reset_pool_cursor": pool_cursor,
                    "curriculum": None,
                    "num_envs": num_envs,
                    "rollout_len": rollout_len,
                    "reset_pool_size": reset_pool_size,
                    "evaluation_protocol_id": protocol_id,
                    "evaluation_protocol_sha256": protocol_sha,
                    "lineage": lineage_hashes(),
                    "dtype": "float32",
                    "static_profile": "v4_2_selected",
                    "parent_class": "EMERGENCY_EXACT_RESUME_FROM_R_E6",
                    "parent_checkpoint": "experiments/competition_native_jax/v4_3_r_e6/ckpt_final",
                    "start_update": start_updates,
                    "start_transitions": start_transitions,
                }
                p = _save_ckpt_atomic(tag, params=params, ema=ema, opt_state=opt_state, meta=meta)
                ckpt_paths.append(str(p))
                print(f"CKPT {tag}", flush=True)

    except KeyboardInterrupt:
        status = "ABORTED_KEYBOARD"
        exit_reason = "SIGINT_KEYBOARDINTERRUPT"

    # final checkpoint
    meta = {
        "update": updates,
        "transitions": transitions,
        "lr": 3e-4,
        "model_rng": [int(x) for x in list(jnp.asarray(key).reshape(-1)[:2])],
        "env_rng_seed_base": 0,
        "reset_pool_seed": pool_seed_u32,
        "reset_pool_cursor": pool_cursor,
        "curriculum": None,
        "num_envs": num_envs,
        "rollout_len": rollout_len,
        "reset_pool_size": reset_pool_size,
        "evaluation_protocol_id": protocol_id,
        "evaluation_protocol_sha256": protocol_sha,
        "lineage": lineage_hashes(),
        "dtype": "float32",
        "static_profile": "v4_2_selected",
        "parent_class": "EMERGENCY_EXACT_RESUME_FROM_R_E6",
        "parent_checkpoint": "experiments/competition_native_jax/v4_3_r_e6/ckpt_final",
        "start_update": start_updates,
        "start_transitions": start_transitions,
        "exit_reason": exit_reason,
    }
    final_path = _save_ckpt_atomic("final", params=params, ema=ema, opt_state=opt_state, meta=meta)
    ckpt_paths.append(str(final_path))
    elapsed = time.perf_counter() - t0
    session_trans = transitions - start_transitions
    tps = session_trans / max(elapsed, 1e-6)

    report = {
        "schema_version": 1,
        "kind": "EMERGENCY_EXACT_RESUME_TRAIN",
        "status": status,
        "exit_reason": exit_reason,
        "start_updates": start_updates,
        "start_transitions": start_transitions,
        "updates": updates,
        "transitions": transitions,
        "session_transitions": session_trans,
        "elapsed_s": elapsed,
        "measured_tps": tps,
        "peak_vram_mib": peak_vram,
        "host_rss_bytes": _rss_bytes(),
        "final_checkpoint": str(final_path),
        "checkpoints": ckpt_paths,
        "last_metrics": {k: float(v) for k, v in last_metrics.items()} if last_metrics else {},
        "device": device,
        **lineage_hashes(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(ROOT / "experiments/manifests/emergency_exact_resume_report.json", report)
    _atomic_write_json(RUNTIME / "programme" / "exact_resume_report.json", report)
    _append_metrics(
        {
            "schema_version": 1,
            "kind": "EMERGENCY_TRAINING_LATEST",
            "updates": updates,
            "transitions": transitions,
            "session_transitions": session_trans,
            "tps": tps,
            "elapsed_s": elapsed,
            "status": status,
            "exit_reason": exit_reason,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    )
    _set_gpu_owner("NONE")
    print(json.dumps({"status": status, "updates": updates, "tps": tps, "final": str(final_path)}, indent=2))
    return 0 if status == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
