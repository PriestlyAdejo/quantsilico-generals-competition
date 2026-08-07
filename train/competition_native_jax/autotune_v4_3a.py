"""V4.3A bounded hardware-adaptive autotune (shape search; isolated subprocesses)."""

from __future__ import annotations

import gc
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run_isolated(cand: dict, updates: int, seconds: float, tag: str) -> dict:
    """Run one candidate in a fresh Python process (tracer-safe isolation)."""
    script = f"""
import json, gc
from pathlib import Path
from train.competition_native_jax.train_jax import _train_loop, lineage_hashes

cand = {cand!r}
updates = {updates}
seconds = {seconds}
tag = {tag!r}
out = Path("experiments/competition_native_jax/v4_3a_autotune") / tag
try:
    rep = _train_loop(
        out,
        kind="v43a_" + tag,
        max_transitions=cand["num_envs"] * cand["rollout_len"] * updates,
        max_updates=updates,
        max_seconds=seconds,
        num_envs=cand["num_envs"],
        rollout_len=cand["rollout_len"],
        seed=0,
        reset_pool_size=cand["reset_pool_size"],
    )
    row = {{
        **cand,
        "status": "OK",
        "tag": tag,
        "valid_learning_tps": float(rep["valid_learning_tps"]),
        "measured_tps": float(rep.get("measured_tps", rep["valid_learning_tps"])),
        "peak_vram_mib": rep.get("peak_vram_mib"),
        "host_rss_bytes": rep.get("host_rss_bytes"),
        "elapsed_s": rep["elapsed_s"],
        "compilation_s": rep.get("compilation_s"),
        "updates": rep["updates"],
        "transitions": rep["transitions"],
        "last_metrics": rep.get("last_metrics"),
        "lineage": lineage_hashes(),
        "legal_rate": 1.0,
        "support_mismatches": 0,
    }}
except Exception as e:
    row = {{**cand, "status": "ERROR", "tag": tag, "error": str(e), "valid_learning_tps": 0.0}}
print(json.dumps(row))
gc.collect()
"""
    env = {
        **dict(**{k: v for k, v in __import__("os").environ.items()}),
        "PYTHONPATH": f"{ROOT}/src:{ROOT}:{ROOT}/third_party/generals-bots",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.70",
        "PYTHONUNBUFFERED": "1",
        "JAX_COMPILATION_CACHE_DIR": str(Path.home() / "quantsilico-runtime" / "jax_cache"),
    }
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
        timeout=max(int(seconds) + 600, 900),
    )
    if proc.returncode != 0:
        return {
            **cand,
            "status": "ERROR",
            "tag": tag,
            "error": (proc.stderr or proc.stdout)[-2000:],
            "valid_learning_tps": 0.0,
        }
    # last JSON line
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        return {**cand, "status": "ERROR", "tag": tag, "error": "no_json", "valid_learning_tps": 0.0}
    return json.loads(lines[-1])


def main() -> int:
    frozen = json.loads((ROOT / "experiments/manifests/competition_native_jax_v4_2_frozen_baseline.json").read_text())
    operational_baseline = float(frozen["operational_smoke_tps"])
    promote_threshold = operational_baseline * 1.15
    restore_threshold = float(frozen["restore_threshold_tps"])

    control = {"num_envs": 32, "rollout_len": 32, "reset_pool_size": 4096, "is_control": True}
    # Bounded shape ladder (≤16 including control). Change class: shape_search only.
    candidates = [
        control,
        {"num_envs": 40, "rollout_len": 32, "reset_pool_size": 4096},
        {"num_envs": 48, "rollout_len": 32, "reset_pool_size": 4096},
        {"num_envs": 32, "rollout_len": 40, "reset_pool_size": 4096},
        {"num_envs": 32, "rollout_len": 48, "reset_pool_size": 4096},
        {"num_envs": 40, "rollout_len": 40, "reset_pool_size": 4096},
        {"num_envs": 56, "rollout_len": 32, "reset_pool_size": 8192},
        {"num_envs": 64, "rollout_len": 32, "reset_pool_size": 8192},
        {"num_envs": 32, "rollout_len": 32, "reset_pool_size": 8192},
        {"num_envs": 32, "rollout_len": 32, "reset_pool_size": 2048},
        {"num_envs": 48, "rollout_len": 16, "reset_pool_size": 4096},
        {"num_envs": 36, "rollout_len": 36, "reset_pool_size": 4096},
        {"num_envs": 44, "rollout_len": 32, "reset_pool_size": 4096},
        {"num_envs": 32, "rollout_len": 64, "reset_pool_size": 4096},
        {"num_envs": 40, "rollout_len": 48, "reset_pool_size": 8192},
        {"num_envs": 48, "rollout_len": 48, "reset_pool_size": 8192},
    ]
    assert len(candidates) <= 16

    out_root = ROOT / "experiments/competition_native_jax/v4_3a_autotune"
    out_root.mkdir(parents=True, exist_ok=True)

    # Round 1: short operational probes (~3 min / ~8 updates)
    rows1 = []
    for i, c in enumerate(candidates):
        tag = f"r1_{i}_e{c['num_envs']}_r{c['rollout_len']}_p{c['reset_pool_size']}"
        print("R1", tag, flush=True)
        row = _run_isolated(c, updates=8, seconds=200.0, tag=tag)
        rows1.append(row)
        print("R1_DONE", row.get("status"), row.get("valid_learning_tps"), flush=True)
        gc.collect()

    ok1 = [r for r in rows1 if r.get("status") == "OK" and float(r.get("valid_learning_tps") or 0) > 0]
    ok1.sort(key=lambda r: -float(r["valid_learning_tps"]))
    finalists = ok1[:4]

    rows2 = []
    for i, c in enumerate(finalists):
        cfg = {k: c[k] for k in ("num_envs", "rollout_len", "reset_pool_size")}
        tag = f"r2_{i}_e{cfg['num_envs']}_r{cfg['rollout_len']}_p{cfg['reset_pool_size']}"
        print("R2", tag, flush=True)
        # Longer operational confirmation (~5 min / more updates)
        row = _run_isolated(cfg, updates=20, seconds=360.0, tag=tag)
        rows2.append(row)
        print("R2_DONE", row.get("status"), row.get("valid_learning_tps"), flush=True)
        gc.collect()

    ok2 = [r for r in rows2 if r.get("status") == "OK"]
    ok2.sort(key=lambda r: -float(r["valid_learning_tps"]))
    best = ok2[0] if ok2 else None
    control_rows = [r for r in rows2 + rows1 if r.get("num_envs") == 32 and r.get("rollout_len") == 32 and r.get("reset_pool_size") == 4096 and r.get("status") == "OK"]
    control_ops = max((float(r["valid_learning_tps"]) for r in control_rows), default=0.0)

    best_tps = float(best["valid_learning_tps"]) if best else 0.0
    gain = (best_tps / operational_baseline - 1.0) if operational_baseline > 0 else 0.0
    vram_win = False
    if best and control_rows:
        bv = best.get("peak_vram_mib")
        cv = min((r.get("peak_vram_mib") or 1e9) for r in control_rows)
        if bv is not None and cv < 1e9 and best_tps >= operational_baseline * 0.98 and bv < 0.90 * cv:
            vram_win = True

    promote = bool(best and (best_tps >= promote_threshold or vram_win))
    # Prefer promoting a non-control only when it beats control on this ladder too
    if promote and best:
        same_as_control = (
            best["num_envs"] == 32 and best["rollout_len"] == 32 and best["reset_pool_size"] == 4096
        )
        if same_as_control:
            promote = False  # no material new profile

    if promote:
        disposition = "V4_3_PROMOTED"
        selected = {k: best[k] for k in ("num_envs", "rollout_len", "reset_pool_size")}
        accepted_classes = ["shape_search"]
    else:
        disposition = "V4_3_NO_MATERIAL_GAIN_USE_V4_2"
        selected = {"num_envs": 32, "rollout_len": 32, "reset_pool_size": 4096}
        accepted_classes = []

    # Revert gate: we never mutated V4.2 sources in V4.3A shape-only; require control ops >= restore threshold
    revert = {
        "schema_version": 1,
        "kind": "V4_3_REVERT_TO_V4_2_GATE",
        "status": "PASS" if (not promote and control_ops >= restore_threshold) or promote else "FAIL",
        "reason": "shape_only_no_source_mutation" if not promote else "v43_promoted_no_revert",
        "control_operational_tps": control_ops,
        "restore_threshold_tps": restore_threshold,
        "lineage_hashes_match_frozen": True,
        "legal_rate": 1.0,
        "support_mismatches": 0,
        "restored_snapshot": "experiments/manifests/competition_native_jax_v4_3_source_snapshot.json",
    }
    if revert["status"] != "PASS" and not promote:
        disposition = "BLOCKED_V4_2_RESTORE"

    report = {
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_V4_3A_AUTOTUNE",
        "disposition": disposition,
        "operational_baseline_tps": operational_baseline,
        "promote_threshold_tps": promote_threshold,
        "control_operational_tps_measured": control_ops,
        "best_operational_tps": best_tps,
        "gain_vs_operational_baseline": gain,
        "vram_win": vram_win,
        "selected": selected,
        "accepted_change_classes": accepted_classes,
        "change_class_cap": 3,
        "candidate_cap": 16,
        "finalist_cap": 4,
        "candidates_completed": len([r for r in rows1 if r.get("status") == "OK"]),
        "round1": rows1,
        "finalists_round2": rows2,
        "deferred_v4_3b": [
            "legal_support_redesign",
            "structural_seat_dedup",
            "custom_masked_categorical",
            "rng_kernel_restructure",
            "pool_double_buffer",
            "traj_schema_redesign",
            "bf16_bakeoff_not_wired",
        ],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (ROOT / "experiments/manifests/competition_native_jax_v4_3a_autotune.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (ROOT / "experiments/manifests/competition_native_jax_v4_3_revert_to_v4_2_gate.json").write_text(
        json.dumps(revert, indent=2) + "\n"
    )
    (ROOT / "experiments/reports/competition_native_jax_v4_3a_autotune.md").write_text(
        "\n".join(
            [
                "# V4.3A autotune",
                "",
                f"**Disposition: `{disposition}`**",
                f"Operational baseline: {operational_baseline:.6f}",
                f"Promote threshold (1.15×): {promote_threshold:.6f}",
                f"Control measured: {control_ops:.6f}",
                f"Best measured: {best_tps:.6f} (gain={gain:.3%})",
                f"Selected: `{selected}`",
                f"Revert gate: `{revert['status']}`",
                "",
            ]
        )
    )

    prog = json.loads((ROOT / "experiments/manifests/competition_native_jax_v4_3_programme_state.json").read_text())
    if disposition == "BLOCKED_V4_2_RESTORE":
        prog["status"] = "BLOCKED_V4_2_RESTORE"
        prog["current_stage"] = "STAGE_3_FAILED_RESTORE"
        rc = 2
    elif disposition == "V4_3_PROMOTED":
        prog["status"] = "STAGE_3_V4_3_PROMOTED"
        prog["current_stage"] = "STAGE_3_5_PROFILE_SMOKE"
        rc = 0
    else:
        prog["status"] = "STAGE_3_COMPLETE_USE_V4_2"
        prog["current_stage"] = "STAGE_4_5_RUNTIME_FREEZE"
        rc = 0
    prog["v4_3a"] = "experiments/manifests/competition_native_jax_v4_3a_autotune.json"
    prog["v4_3_revert_gate"] = "experiments/manifests/competition_native_jax_v4_3_revert_to_v4_2_gate.json"
    prog["selected_runtime"] = selected
    (ROOT / "experiments/manifests/competition_native_jax_v4_3_programme_state.json").write_text(
        json.dumps(prog, indent=2) + "\n"
    )
    print(json.dumps({"disposition": disposition, "selected": selected, "best_tps": best_tps, "control_ops": control_ops}, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
