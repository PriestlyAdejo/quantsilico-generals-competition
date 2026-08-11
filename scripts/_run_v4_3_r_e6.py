"""R-E.6 short daytime PPO: compatible cold restart + full training-state checkpoints."""

from __future__ import annotations

import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

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
    runtime_dir,
    save_training_checkpoint,
    save_tree,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    runtime = json.loads((ROOT / "experiments/manifests/daytime_runtime_selected.json").read_text())
    freeze = json.loads(
        (ROOT / "experiments/manifests/competition_native_jax_daytime_eval_protocol_freeze.json").read_text()
    )
    prog = json.loads((ROOT / "experiments/manifests/competition_native_jax_v4_3_programme_state.json").read_text())
    ckpt_status = prog.get("checkpoint_roundtrip_status")
    assert ckpt_status == "CHECKPOINT_EXACT_CONTINUATION_PASS"

    sel = runtime["selected"]
    num_envs = int(sel["num_envs"])
    rollout_len = int(sel["rollout_len"])
    reset_pool_size = int(sel["reset_pool_size"])
    bud = runtime["budgets"]["r_e6"]
    max_updates = int(bud["max_complete_updates"])
    max_transitions = int(bud["transition_budget"])
    max_seconds = float(bud["max_seconds"])
    protocol_id = freeze["evaluation_protocol_id"]
    protocol_sha = freeze["evaluation_protocol_sha256"]

    out_dir = ROOT / "experiments/competition_native_jax/v4_3_r_e6"
    out_dir.mkdir(parents=True, exist_ok=True)
    rt = runtime_dir("r_e6")
    stop_path = out_dir / "STOP_REQUEST"

    launch = {
        "schema_version": 1,
        "kind": "R_E6_LAUNCH_MANIFEST",
        "parent_class": "R_E6_PARENT_COMPATIBLE_COLD_RESTART",
        "selected": sel,
        "max_complete_updates": max_updates,
        "transition_budget": max_transitions,
        "max_seconds": max_seconds,
        "evaluation_protocol_id": protocol_id,
        "evaluation_protocol_sha256": protocol_sha,
        "seed": 0,
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "overnight_execution_authorized": False,
        "portal_upload_authorized": False,
    }
    (ROOT / "experiments/manifests/competition_native_jax_v4_3_r_e6_launch.json").write_text(
        json.dumps(launch, indent=2) + "\n"
    )
    owned = {
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_OWNED_PROCESSES",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "jobs": [
            {
                "id": "v43_re6",
                "kind": "R_E6_SHORT",
                "status": "RUNNING",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
        ],
    }
    (ROOT / "experiments/manifests/competition_native_jax_owned_processes.json").write_text(
        json.dumps(owned, indent=2) + "\n"
    )

    device = detect_jax_device()
    print("R-E.6 start", json.dumps({"sel": sel, "max_updates": max_updates, "device": device}), flush=True)

    t_compile0 = time.perf_counter()
    key = jax.random.PRNGKey(0)  # same original init lineage as R-E.5 cold path
    params = jax.device_put(init_params(key))
    _ = forward(params, jnp.zeros((N_SPATIAL, 21, 21)), jnp.zeros((N_GLOBAL,)))
    compile_s = time.perf_counter() - t_compile0

    ema = params
    optimizer = make_optimizer(3e-4)
    opt_state = optimizer.init(params)
    key_pool, key = jax.random.split(key)
    pool_seed_u32 = int(key_pool[0]) if hasattr(key_pool, "__getitem__") else 0
    reset_pool = build_competition_reset_pool(key_pool, reset_pool_size)
    jax.block_until_ready(jax.tree_util.tree_leaves(reset_pool)[0])
    pool_refresh_every = 10
    pool_cursor = 0

    # warm compile path
    warm = collect_selfplay_batch(
        params,
        num_envs=num_envs,
        rollout_len=rollout_len,
        seed=0,
        reset_pool_size=reset_pool_size,
        pool=reset_pool,
    )
    jax.block_until_ready(warm["rewards"])
    Tw, Nw = warm["rewards"].shape
    warm_flat = {
        "spatial": warm["spatial"].reshape(Tw * Nw, *warm["spatial"].shape[2:]),
        "global": warm["global"].reshape(Tw * Nw, *warm["global"].shape[2:]),
        "mask": warm["mask"].reshape(Tw * Nw, -1),
        "actions": warm["actions"].reshape(Tw * Nw),
        "old_logp": warm["old_logp"].reshape(Tw * Nw),
    }
    warm_vals = jnp.concatenate([warm["values"], warm["bootstrap_values"][None, :]], axis=0)
    warm_adv, warm_ret = gae_advantages_batch_jit(warm["rewards"], warm_vals, warm["dones"])
    warm_flat["advantages"] = warm_adv.reshape(Tw * Nw)
    warm_flat["returns"] = warm_ret.reshape(Tw * Nw)
    params, opt_state, _ = ppo_update(params, opt_state, optimizer, warm_flat)
    ema = ema_update(ema, params)

    milestones = {0, max(1, max_updates // 4), max(1, max_updates // 2), max(1, (3 * max_updates) // 4), max_updates}
    transitions = 0
    updates = 0
    peak_vram = _vram_used_mib() or 0.0
    t0 = time.perf_counter()
    last_metrics: dict = {}
    status = "COMPLETED"
    exit_reason = "BUDGET_OR_TIME"
    ckpt_paths: list[str] = []

    def _save_ckpt(tag: str) -> str:
        nonlocal pool_cursor
        ckpt_dir = out_dir / f"ckpt_{tag}"
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
            "parent_class": "R_E6_PARENT_COMPATIBLE_COLD_RESTART",
        }
        save_training_checkpoint(ckpt_dir, params=params, ema=ema, opt_state=opt_state, meta=meta)
        # also mirror raw/ema into runtime for convenience
        save_tree(rt / f"r_e6_{tag}_raw.npz", params)
        save_tree(rt / f"r_e6_{tag}_ema.npz", ema)
        rel = str(ckpt_dir.relative_to(ROOT)).replace("\\", "/")
        ckpt_paths.append(rel)
        print(f"CKPT {tag} updates={updates} transitions={transitions}", flush=True)
        return rel

    try:
        while transitions < max_transitions and updates < max_updates and (time.perf_counter() - t0) < max_seconds:
            if stop_path.exists():
                status = "ABORTED_STOP_REQUEST"
                exit_reason = "STOP_REQUEST_FILE"
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
            if updates in milestones or updates == 1:
                _save_ckpt(f"u{updates}")
            if updates == 1 or updates % 10 == 0:
                hb = {
                    "transitions": transitions,
                    "updates": updates,
                    "elapsed_s": time.perf_counter() - t0,
                    "tps": transitions / max(time.perf_counter() - t0, 1e-6),
                    "ts": time.time(),
                }
                (rt / "heartbeat.json").write_text(json.dumps(hb))
                shutil.copy2(rt / "heartbeat.json", out_dir / "heartbeat.json")
                print(
                    f"HB updates={updates}/{max_updates} transitions={transitions} tps={hb['tps']:.2f}",
                    flush=True,
                )
    except KeyboardInterrupt:
        status = "ABORTED_KEYBOARD"
        exit_reason = "SIGINT_KEYBOARDINTERRUPT"

    v_end = _vram_used_mib()
    if v_end is not None:
        peak_vram = max(peak_vram, v_end)
    elapsed = time.perf_counter() - t0
    final_ckpt = _save_ckpt("final")
    tps = transitions / max(elapsed, 1e-6)

    report = {
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_R_E6_TRAIN",
        "status": status,
        "exit_reason": exit_reason,
        "transitions": transitions,
        "updates": updates,
        "elapsed_s": elapsed,
        "compilation_s": compile_s,
        "measured_tps": tps,
        "valid_learning_tps": tps,
        "peak_vram_mib": peak_vram,
        "host_rss_bytes": _rss_bytes(),
        "num_envs": num_envs,
        "rollout_len": rollout_len,
        "reset_pool_size": reset_pool_size,
        "device": device,
        "final_checkpoint": final_ckpt,
        "checkpoints": ckpt_paths,
        "evaluation_protocol_id": protocol_id,
        "evaluation_protocol_sha256": protocol_sha,
        "parent_class": "R_E6_PARENT_COMPATIBLE_COLD_RESTART",
        "last_metrics": {k: float(v) for k, v in last_metrics.items()} if last_metrics else {},
        "limits": {
            "max_transitions": max_transitions,
            "max_updates": max_updates,
            "max_seconds": max_seconds,
        },
        **lineage_hashes(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (ROOT / "experiments/manifests/competition_native_jax_v4_3_r_e6.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (out_dir / "r_e6_report.json").write_text(json.dumps(report, indent=2) + "\n")

    prog["status"] = "STAGE_6_COMPLETE" if status == "COMPLETED" else f"STAGE_6_{status}"
    prog["current_stage"] = "STAGE_7_DAYTIME_EVAL"
    prog["r_e6"] = "experiments/manifests/competition_native_jax_v4_3_r_e6.json"
    prog["r_e6_final_checkpoint"] = final_ckpt
    (ROOT / "experiments/manifests/competition_native_jax_v4_3_programme_state.json").write_text(
        json.dumps(prog, indent=2) + "\n"
    )
    print(json.dumps({"status": status, "updates": updates, "tps": tps, "final_ckpt": final_ckpt}, indent=2))
    return 0 if status == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
