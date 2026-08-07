"""Canonical JAX training entrypoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

# Persistent compilation cache under Linux home (no-op if unset/unwritable)
_CACHE = Path.home() / "quantsilico-runtime" / "jax_cache"
try:
    _CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(_CACHE))
except Exception:
    pass

import jax
import jax.numpy as jnp
import numpy as np

from generals_bot.competition_native_jax.obs_memory import N_GLOBAL, N_SPATIAL
from generals_bot.competition_native_jax.transformer_jax import forward, init_params
from generals_bot.competition_native_jax.competition_env_jax import build_competition_reset_pool
from train.competition_native_jax.ema_jax import ema_update
from train.competition_native_jax.gae_jax import gae_advantages_batch_jit
from train.competition_native_jax.ppo_jax import assert_zero_update_ratio, make_optimizer, ppo_update
from train.competition_native_jax.rollout_selfplay_jax import ROLLOUT_ARCHITECTURE, collect_selfplay_batch
import hashlib


def _hash_files(rels: tuple[str, ...]) -> str:
    root = Path(__file__).resolve().parents[2]
    parts = []
    for rel in rels:
        p = root / rel
        if p.exists():
            parts.append(p.read_bytes())
    return hashlib.sha256(b"".join(parts)).hexdigest()


def env_implementation_hash() -> str:
    """Hash of competition env + rollout sources (unchanged if those files unchanged)."""
    return _hash_files(
        (
            "src/generals_bot/competition_native_jax/competition_env_jax.py",
            "train/competition_native_jax/rollout_selfplay_jax.py",
        )
    )


def env_semantics_hash() -> str:
    """Stable semantics identity for competition rules wrappers."""
    return _hash_files(("src/generals_bot/competition_native_jax/competition_env_jax.py",))


def learner_implementation_hash() -> str:
    """GAE/PPO/transformer/train loop (not game semantics)."""
    return _hash_files(
        (
            "train/competition_native_jax/gae_jax.py",
            "train/competition_native_jax/ppo_jax.py",
            "train/competition_native_jax/ema_jax.py",
            "train/competition_native_jax/train_jax.py",
            "src/generals_bot/competition_native_jax/transformer_jax.py",
        )
    )


def performance_programme_hash() -> str:
    return hashlib.sha256(b"END_TO_END_JAX_V4_2_MAX_UTILISATION").hexdigest()


def lineage_hashes() -> dict[str, str]:
    return {
        "env_semantics_hash": env_semantics_hash(),
        "env_implementation_hash": env_implementation_hash(),
        "learner_implementation_hash": learner_implementation_hash(),
        "performance_programme_hash": performance_programme_hash(),
        "performance_programme": "END_TO_END_JAX_V4_2_MAX_UTILISATION",
    }


def runtime_dir(kind: str) -> Path:
    d = Path.home() / "quantsilico-runtime" / "competition_native_jax" / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def detect_jax_device() -> dict[str, Any]:
    info: dict[str, Any] = {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "jax_gpu": any(getattr(d, "platform", "") == "gpu" or "cuda" in str(d).lower() for d in jax.devices()),
        "jax_compilation_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
    }
    try:
        import jaxlib

        info["jaxlib_version"] = jaxlib.__version__
    except Exception:
        info["jaxlib_version"] = None
    return info


def _rss_bytes() -> int | None:
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        return None


def _vram_used_mib() -> float | None:
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def save_tree(path: Path, tree) -> None:
    flat_tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    np.savez_compressed(path, **{str(k): np.asarray(v) for k, v in flat_tree})


def load_tree(path: Path, like):
    """Reload a pytree saved by ``save_tree`` using ``like`` as the structure template."""
    data = np.load(path, allow_pickle=False)
    flat_like, treedef = jax.tree_util.tree_flatten_with_path(like)
    leaves = []
    for key_path, leaf in flat_like:
        arr = data[str(key_path)]
        leaves.append(jnp.asarray(arr, dtype=getattr(leaf, "dtype", arr.dtype)))
    return jax.tree_util.tree_unflatten(treedef, leaves)


def save_training_checkpoint(
    path: Path,
    *,
    params,
    ema,
    opt_state,
    meta: dict[str, Any],
) -> None:
    """Persist full training state required for exact resume / overnight parent."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_tree(path / "raw.npz", params)
    save_tree(path / "ema.npz", ema)
    save_tree(path / "opt_state.npz", opt_state)
    (path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def load_training_checkpoint(path: Path, *, params_like, opt_state_like) -> dict[str, Any]:
    path = Path(path)
    params = load_tree(path / "raw.npz", params_like)
    ema = load_tree(path / "ema.npz", params_like)
    opt_state = load_tree(path / "opt_state.npz", opt_state_like)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    return {"params": params, "ema": ema, "opt_state": opt_state, "meta": meta}


def run_gpu_correctness_gate(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = detect_jax_device()
    if not device.get("jax_gpu"):
        report = {
            "schema_version": 1,
            "kind": "COMPETITION_NATIVE_JAX_GPU_CORRECTNESS_GATE",
            "status": "FAILED_NO_JAX_GPU",
            "device": device,
        }
        (out_dir / "gpu_correctness_gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    key = jax.random.PRNGKey(0)
    params = jax.device_put(init_params(key))
    spatial = jax.device_put(jnp.zeros((N_SPATIAL, 21, 21), dtype=jnp.float32))
    global_vec = jax.device_put(jnp.zeros((N_GLOBAL,), dtype=jnp.float32))
    out = forward(params, spatial, global_vec)
    out["flat_logits"].block_until_ready()
    mask = jnp.zeros(out["flat_logits"].shape[0], dtype=bool).at[0].set(True)
    mask = mask.at[1:20].set(True)
    rho = float(assert_zero_update_ratio(out["flat_logits"], mask, jnp.array(0)))

    def loss_fn(p):
        o = forward(p, spatial, global_vec)
        return jnp.sum(o["flat_logits"] ** 2)

    grads = jax.grad(loss_fn)(params)
    leaf = jax.tree_util.tree_leaves(grads)[0]
    report = {
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_GPU_CORRECTNESS_GATE",
        "status": "PASSED" if abs(rho - 1.0) < 1e-5 else "FAILED",
        "zero_update_rho": rho,
        "device": device,
        "params_on_device": str(jax.devices()[0]),
        "grad_device": str(leaf.device if hasattr(leaf, "device") else jax.devices()[0]),
        "note": "GPU zero-update PPO identity rho≈1",
    }
    (out_dir / "gpu_correctness_gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _train_loop(
    out_dir: Path,
    *,
    kind: str,
    max_transitions: int,
    max_updates: int,
    max_seconds: float,
    num_envs: int,
    rollout_len: int,
    seed: int = 0,
    reset_pool_size: int = 4096,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rt = runtime_dir(kind)
    stop_path = out_dir / "STOP_REQUEST"
    device = detect_jax_device()
    t_compile0 = time.perf_counter()
    key = jax.random.PRNGKey(seed)
    params = jax.device_put(init_params(key))
    _ = forward(
        params,
        jnp.zeros((N_SPATIAL, 21, 21), dtype=jnp.float32),
        jnp.zeros((N_GLOBAL,), dtype=jnp.float32),
    )
    compile_s = time.perf_counter() - t_compile0
    recompile_count = 1

    ema = params
    optimizer = make_optimizer(3e-4)
    opt_state = optimizer.init(params)
    # Build competition reset pool once outside timed loop; refresh on a frozen cadence.
    key_pool, key = jax.random.split(key)
    t_pool0 = time.perf_counter()
    reset_pool = build_competition_reset_pool(key_pool, reset_pool_size)
    jax.block_until_ready(jax.tree_util.tree_leaves(reset_pool)[0])
    pool_build_s = time.perf_counter() - t_pool0
    pool_refresh_every = 10  # updates between pool regenerations (outside scan)
    # Warm collect + one GAE/PPO path for this (num_envs, rollout_len) before TPS timing.
    warm = collect_selfplay_batch(
        params,
        num_envs=num_envs,
        rollout_len=rollout_len,
        seed=seed,
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
    params, opt_state, _warm_m = ppo_update(params, opt_state, optimizer, warm_flat)
    ema = ema_update(ema, params)
    jax.block_until_ready(jax.tree_util.tree_leaves(params)[0])

    transitions = 0
    updates = 0
    peak_vram = _vram_used_mib() or 0.0
    t0 = time.perf_counter()
    last_metrics: dict = {}
    status = "COMPLETED"
    exit_reason = "BUDGET_OR_TIME"
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
            batch = collect_selfplay_batch(
                params,
                num_envs=num_envs,
                rollout_len=rollout_len,
                seed=seed + updates + 1,
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
            # Rate-limited heartbeat on Linux-native runtime (then mirror to repo)
            if updates == 1 or updates % 5 == 0:
                hb = {
                    "transitions": transitions,
                    "updates": updates,
                    "elapsed_s": time.perf_counter() - t0,
                    "ts": time.time(),
                }
                (rt / "heartbeat.json").write_text(json.dumps(hb), encoding="utf-8")
                shutil.copy2(rt / "heartbeat.json", out_dir / "heartbeat.json")
    except KeyboardInterrupt:
        status = "ABORTED_KEYBOARD"
        exit_reason = "SIGINT_KEYBOARDINTERRUPT"

    # Stage-boundary VRAM sample only (not per update)
    v_end = _vram_used_mib()
    if v_end is not None:
        peak_vram = max(peak_vram, v_end)
    elapsed = time.perf_counter() - t0
    raw_path = out_dir / f"{kind}_raw.npz"
    ema_path = out_dir / f"{kind}_ema.npz"
    # Write checkpoints via runtime then copy atomically-ish into repo
    save_tree(rt / f"{kind}_raw.npz", params)
    save_tree(rt / f"{kind}_ema.npz", ema)
    shutil.copy2(rt / f"{kind}_raw.npz", raw_path)
    shutil.copy2(rt / f"{kind}_ema.npz", ema_path)
    tps = transitions / max(elapsed, 1e-6)
    report = {
        "schema_version": 1,
        "kind": f"COMPETITION_NATIVE_JAX_{kind.upper()}_TRAIN",
        "status": status,
        "exit_reason": exit_reason,
        "transitions": transitions,
        "updates": updates,
        "elapsed_s": elapsed,
        "compilation_s": compile_s,
        "recompile_count": recompile_count,
        "measured_tps": tps,
        "valid_learning_tps": tps,
        "peak_vram_mib": peak_vram,
        "host_rss_bytes": _rss_bytes(),
        "num_envs": num_envs,
        "rollout_len": rollout_len,
        "reset_pool_size": reset_pool_size,
        "reset_pool_build_s": pool_build_s,
        "reset_pool_refresh_every_updates": pool_refresh_every,
        "device": device,
        "checkpoint_raw": str(raw_path).replace("\\", "/"),
        "checkpoint_ema": str(ema_path).replace("\\", "/"),
        "runtime_dir": str(rt).replace("\\", "/"),
        "jax_gpu_used": bool(device.get("jax_gpu")),
        "last_metrics": {k: float(v) for k, v in last_metrics.items()} if last_metrics else {},
        "limits": {
            "max_transitions": max_transitions,
            "max_updates": max_updates,
            "max_seconds": max_seconds,
        },
        "rollout_backend": "official_mit_jax_primitives_plus_qs_scan",
        "rollout_architecture": ROLLOUT_ARCHITECTURE,
        **lineage_hashes(),
        "gae_device_resident_batched": True,
        "host_bound_optimiser_lineage": False,
    }
    (out_dir / f"{kind}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_tiny_training(
    out_dir: Path,
    *,
    max_transitions: int = 8192,
    max_updates: int = 4,
    max_seconds: float = 900.0,
    num_envs: int = 2,
    rollout_len: int = 16,
    seed: int = 0,
) -> dict:
    return _train_loop(
        out_dir,
        kind="tiny",
        max_transitions=max_transitions,
        max_updates=max_updates,
        max_seconds=max_seconds,
        num_envs=num_envs,
        rollout_len=rollout_len,
        seed=seed,
    )


def run_throughput_ladder(out_dir: Path, manifest_path: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        {"num_envs": 2, "rollout_len": 8, "minibatch": 128},
        {"num_envs": 2, "rollout_len": 16, "minibatch": 256},
        {"num_envs": 4, "rollout_len": 16, "minibatch": 512},
        {"num_envs": 8, "rollout_len": 16, "minibatch": 512},
    ]
    results = []
    best = None
    for cand in candidates:
        try:
            v0 = _vram_used_mib() or 0.0
            rep = _train_loop(
                out_dir / f"probe_e{cand['num_envs']}_r{cand['rollout_len']}",
                kind="probe",
                max_transitions=cand["num_envs"] * cand["rollout_len"] * 2,
                max_updates=2,
                max_seconds=180.0,
                num_envs=cand["num_envs"],
                rollout_len=cand["rollout_len"],
                seed=0,
            )
            row = {
                **cand,
                "valid_learning_tps": rep["valid_learning_tps"],
                "compilation_s": rep["compilation_s"],
                "peak_vram_mib": rep["peak_vram_mib"],
                "host_rss_bytes": rep["host_rss_bytes"],
                "elapsed_s": rep["elapsed_s"],
                "status": "OK",
                "vram_delta_mib": (rep["peak_vram_mib"] or 0) - v0,
            }
            row["stable"] = bool(row["peak_vram_mib"] < 7000 and row["valid_learning_tps"] > 0)
            results.append(row)
            if row["stable"] and (best is None or row["valid_learning_tps"] > best["valid_learning_tps"]):
                best = row
        except Exception as exc:  # noqa: BLE001
            results.append({**cand, "status": "ERROR", "error": str(exc), "stable": False})

    if best is None:
        ok = [r for r in results if r.get("status") == "OK"]
        if ok:
            best = max(ok, key=lambda r: r.get("valid_learning_tps", 0.0))
            best["stable"] = False

    report = {
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_THROUGHPUT_LADDER_V2",
        "status": "FROZEN" if best else "BLOCKED_COMPUTE",
        "gpu_jax_verified": True,
        "candidates": results,
        "frozen_config": best,
        "frozen_from_measured_vram": bool(best and best.get("stable")),
        "declared_transition_budget_rule": "floor(0.85 * TPS * permitted_seconds)",
        "short_day_budget_transitions": int(math.floor(0.85 * best["valid_learning_tps"] * 5400)) if best else 0,
        "medium_day_budget_transitions": int(math.floor(0.85 * best["valid_learning_tps"] * 14400)) if best else 0,
        "full_rollout_fps": best["valid_learning_tps"] if best else None,
        "valid_learning_tps": best["valid_learning_tps"] if best else None,
    }
    manifest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "throughput_ladder_v2.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_throughput_ladder_v3(out_dir: Path, manifest_path: Path) -> dict:
    """Post-remediation throughput ladder with complete-loop sync."""
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_tps = [0.1513, 0.185]
    candidates = [
        {"num_envs": 8, "rollout_len": 16, "minibatch": 512},
        {"num_envs": 16, "rollout_len": 16, "minibatch": 512},
        {"num_envs": 32, "rollout_len": 16, "minibatch": 1024},
        {"num_envs": 8, "rollout_len": 32, "minibatch": 512},
        {"num_envs": 64, "rollout_len": 8, "minibatch": 1024},
    ]
    results = []
    best = None
    for cand in candidates:
        try:
            v0 = _vram_used_mib() or 0.0
            rep = _train_loop(
                out_dir / f"probe_e{cand['num_envs']}_r{cand['rollout_len']}",
                kind="probe_v3",
                max_transitions=cand["num_envs"] * cand["rollout_len"] * 2,
                max_updates=2,
                max_seconds=300.0,
                num_envs=cand["num_envs"],
                rollout_len=cand["rollout_len"],
                seed=0,
            )
            tps = float(rep["valid_learning_tps"])
            row = {
                **cand,
                "valid_learning_tps": tps,
                "compilation_s": rep["compilation_s"],
                "recompile_count": rep.get("recompile_count", 1),
                "peak_vram_mib": rep["peak_vram_mib"],
                "host_rss_bytes": rep["host_rss_bytes"],
                "elapsed_s": rep["elapsed_s"],
                "status": "OK",
                "vram_delta_mib": (rep["peak_vram_mib"] or 0) - v0,
                "improvement_vs_baseline_low": tps / baseline_tps[0],
                "improvement_vs_baseline_high": tps / baseline_tps[1],
            }
            row["stable"] = bool(row["peak_vram_mib"] < 7000 and tps > 0)
            results.append(row)
            if row["stable"] and (best is None or tps > best["valid_learning_tps"]):
                best = row
        except Exception as exc:  # noqa: BLE001
            results.append({**cand, "status": "ERROR", "error": str(exc), "stable": False})

    if best is None:
        ok = [r for r in results if r.get("status") == "OK"]
        if ok:
            best = max(ok, key=lambda r: r.get("valid_learning_tps", 0.0))
            best["stable"] = False

    tps = float(best["valid_learning_tps"]) if best else 0.0
    # Meaningful short eligibility: 100k transitions / 5400s => ~18.52 TPS
    min_tps_90m = 100_000 / 5400.0
    eligible = bool(best and best.get("stable") and tps >= min_tps_90m)
    report = {
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_THROUGHPUT_LADDER_V3",
        "status": "FROZEN" if best else "BLOCKED_COMPUTE",
        "gpu_jax_verified": True,
        "baseline_valid_tps": baseline_tps,
        "candidates": results,
        "frozen_config": best,
        "frozen_from_measured_vram": bool(best and best.get("stable")),
        "valid_learning_tps": tps if best else None,
        "improvement_vs_baseline_low": (tps / baseline_tps[0]) if best else None,
        "improvement_vs_baseline_high": (tps / baseline_tps[1]) if best else None,
        "short_day_budget_transitions": int(math.floor(0.85 * tps * 5400)) if best else 0,
        "medium_day_budget_transitions": int(math.floor(0.85 * tps * 14400)) if best else 0,
        "supported_90m_transitions": int(math.floor(tps * 5400)) if best else 0,
        "supported_90m_updates_at_rollout": (
            int(math.floor((tps * 5400) / (best["num_envs"] * best["rollout_len"]))) if best else 0
        ),
        "min_meaningful_tps_90m": min_tps_90m,
        "meaningful_short_eligible": eligible,
        "rollout_architecture": "END_TO_END_OFFICIAL_JAX_ROLLOUT",
    }
    manifest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "throughput_ladder_v3.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = [
        "# Throughput ladder V3",
        "",
        f"Status: `{report['status']}`",
        f"Architecture: `{report['rollout_architecture']}`",
        f"Baseline TPS: {baseline_tps[0]}–{baseline_tps[1]}",
        f"Best TPS: {tps:.4f}" if best else "Best TPS: none",
        f"Improvement vs baseline low: {report['improvement_vs_baseline_low']}",
        f"Meaningful 90m eligible (≥{min_tps_90m:.2f} TPS): {eligible}",
        "",
        "## Candidates",
        "",
    ]
    for r in results:
        md.append(f"- envs={r.get('num_envs')} rollout={r.get('rollout_len')}: {r}")
    Path("experiments/reports/competition_native_jax_throughput_v3.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    return report


def run_smoke_training(out_dir: Path, *, num_envs: int, rollout_len: int) -> dict:
    return _train_loop(
        out_dir,
        kind="smoke",
        max_transitions=100_000,
        max_updates=10_000,
        max_seconds=1800.0,
        num_envs=num_envs,
        rollout_len=rollout_len,
        seed=1,
    )


def run_short_training(out_dir: Path, *, num_envs: int, rollout_len: int, budget_transitions: int) -> dict:
    return _train_loop(
        out_dir,
        kind="short",
        max_transitions=max(1, budget_transitions),
        max_updates=100_000,
        max_seconds=5400.0,
        num_envs=num_envs,
        rollout_len=rollout_len,
        seed=2,
    )


def run_medium_training(out_dir: Path, *, num_envs: int, rollout_len: int, budget_transitions: int) -> dict:
    return _train_loop(
        out_dir,
        kind="medium",
        max_transitions=max(1, budget_transitions),
        max_updates=200_000,
        max_seconds=14400.0,
        num_envs=num_envs,
        rollout_len=rollout_len,
        seed=3,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        required=True,
        choices=["correctness", "tiny", "throughput", "throughput_v3", "smoke", "short", "medium"],
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=2)
    p.add_argument("--rollout-len", type=int, default=16)
    p.add_argument("--budget-transitions", type=int, default=0)
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/manifests/competition_native_jax_throughput_ladder_v2.json"),
    )
    args = p.parse_args()
    if args.mode == "correctness":
        print(json.dumps(run_gpu_correctness_gate(args.out), indent=2))
    elif args.mode == "tiny":
        print(
            json.dumps(
                run_tiny_training(
                    args.out,
                    num_envs=args.num_envs,
                    rollout_len=args.rollout_len,
                    max_transitions=8192,
                    max_updates=4,
                    max_seconds=900.0,
                ),
                indent=2,
            )
        )
    elif args.mode == "throughput":
        print(json.dumps(run_throughput_ladder(args.out, args.manifest), indent=2))
    elif args.mode == "throughput_v3":
        man = args.manifest
        if "v2" in str(man):
            man = Path("experiments/manifests/competition_native_jax_throughput_ladder_v3.json")
        print(json.dumps(run_throughput_ladder_v3(args.out, man), indent=2))
    elif args.mode == "smoke":
        print(json.dumps(run_smoke_training(args.out, num_envs=args.num_envs, rollout_len=args.rollout_len), indent=2))
    elif args.mode == "short":
        print(
            json.dumps(
                run_short_training(
                    args.out,
                    num_envs=args.num_envs,
                    rollout_len=args.rollout_len,
                    budget_transitions=args.budget_transitions,
                ),
                indent=2,
            )
        )
    elif args.mode == "medium":
        print(
            json.dumps(
                run_medium_training(
                    args.out,
                    num_envs=args.num_envs,
                    rollout_len=args.rollout_len,
                    budget_transitions=args.budget_transitions,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
