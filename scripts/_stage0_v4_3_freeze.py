"""Stage 0: archive plan metadata + freeze V4.2/V4.3 programme manifests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RUNTIME_FILES = [
    "src/generals_bot/competition_native_jax/competition_env_jax.py",
    "src/generals_bot/competition_native_jax/transformer_jax.py",
    "src/generals_bot/competition_native_jax/obs_memory.py",
    "src/generals_bot/competition_native_jax/static_geometry_jax.py",
    "train/competition_native_jax/rollout_selfplay_jax.py",
    "train/competition_native_jax/gae_jax.py",
    "train/competition_native_jax/ppo_jax.py",
    "train/competition_native_jax/ema_jax.py",
    "train/competition_native_jax/train_jax.py",
]


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def lineage_h(rels: tuple[str, ...]) -> str:
    parts: list[bytes] = []
    for rel in rels:
        p = ROOT / rel
        parts.append(p.read_bytes() if p.exists() else b"")
    return hashlib.sha256(b"".join(parts)).hexdigest()


def main() -> None:
    files: dict[str, object] = {}
    agg = hashlib.sha256()
    for rel in RUNTIME_FILES:
        p = ROOT / rel
        if not p.exists():
            files[rel] = {"exists": False}
            continue
        data = p.read_bytes()
        h = hashlib.sha256(data).hexdigest()
        files[rel] = {"exists": True, "sha256": h, "bytes": len(data)}
        agg.update(rel.encode())
        agg.update(b"\0")
        agg.update(data)

    env_impl = lineage_h(
        (
            "src/generals_bot/competition_native_jax/competition_env_jax.py",
            "train/competition_native_jax/rollout_selfplay_jax.py",
        )
    )
    env_sem = lineage_h(("src/generals_bot/competition_native_jax/competition_env_jax.py",))
    learn = lineage_h(
        (
            "train/competition_native_jax/gae_jax.py",
            "train/competition_native_jax/ppo_jax.py",
            "train/competition_native_jax/ema_jax.py",
            "train/competition_native_jax/train_jax.py",
            "src/generals_bot/competition_native_jax/transformer_jax.py",
        )
    )

    smoke = json.loads((ROOT / "experiments/manifests/competition_native_jax_v4_2_smoke_r_e5.json").read_text())
    ladder = json.loads((ROOT / "experiments/manifests/competition_native_jax_throughput_ladder_v4_2.json").read_text())
    systems = json.loads(
        (ROOT / "experiments/manifests/competition_native_jax_v4_2_systems_promotion_gate.json").read_text()
    )

    plan_path = "plans/v4_3_daytime_to_package_5333a24e.plan.md"
    plan_sha = hashlib.sha256((ROOT / plan_path).read_bytes()).hexdigest()
    auth_ts = datetime.now(timezone.utc).isoformat()

    clean = float(systems["valid_learning_tps"])
    operational = float(smoke["valid_learning_tps"])
    restore = 0.90 * operational

    reconstructed = (
        env_impl == smoke["env_implementation_hash"]
        and learn == smoke["learner_implementation_hash"]
        and env_sem == smoke["env_semantics_hash"]
    )

    training_config = {
        "num_envs": 32,
        "rollout_len": 32,
        "reset_pool_size": 4096,
        "selected_from": "v4_2_systems_promotion_gate",
    }
    training_config_sha = hashlib.sha256(json.dumps(training_config, sort_keys=True).encode()).hexdigest()

    source_manifest = {
        "schema_version": 1,
        "kind": "DAYTIME_SOURCE_SNAPSHOT_MANIFEST",
        "selection": (
            "CONTENT_ADDRESSED_WORKTREE_MATCHING_R_E5_SMOKE"
            if reconstructed
            else "CONTENT_ADDRESSED_CLEAN_CURRENT"
        ),
        "r_e5_source_status": (
            "R_E5_SOURCE_RECONSTRUCTED" if reconstructed else "R_E5_SOURCE_NOT_RECONSTRUCTABLE"
        ),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["branch", "--show-current"]),
        "dirty_worktree": True,
        "source_manifest_sha": agg.hexdigest(),
        "files": files,
        "env_semantics_hash": env_sem,
        "env_implementation_hash": env_impl,
        "learner_implementation_hash": learn,
        "training_config_sha": training_config_sha,
        "selected_runtime": training_config,
    }
    (ROOT / "experiments/manifests/competition_native_jax_v4_3_source_snapshot.json").write_text(
        json.dumps(source_manifest, indent=2) + "\n"
    )

    frozen = {
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_V4_2_FROZEN_BASELINE",
        "classification": "PROVISIONAL_SYSTEMS_PROMOTION_READY_PENDING_V4_2_STAGE_3B",
        "selected": {"num_envs": 32, "rollout_len": 32, "reset_pool_size": 4096},
        "clean_benchmark_tps": clean,
        "operational_smoke_tps": operational,
        "restore_threshold_tps": restore,
        "ladder_env_implementation_hash": ladder["report"]["env_implementation_hash"],
        "smoke_env_implementation_hash": smoke["env_implementation_hash"],
        "selected_env_implementation_hash": env_impl,
        "selected_env_semantics_hash": env_sem,
        "selected_learner_implementation_hash": learn,
        "note": (
            "Rollback and budgets use operational_smoke_tps / restore_threshold_tps, "
            "never clean_benchmark_tps"
        ),
        "smoke_manifest": "experiments/manifests/competition_native_jax_v4_2_smoke_r_e5.json",
        "ladder_manifest": "experiments/manifests/competition_native_jax_throughput_ladder_v4_2.json",
        "systems_gate": "experiments/manifests/competition_native_jax_v4_2_systems_promotion_gate.json",
        "source_snapshot": "experiments/manifests/competition_native_jax_v4_3_source_snapshot.json",
    }
    (ROOT / "experiments/manifests/competition_native_jax_v4_2_frozen_baseline.json").write_text(
        json.dumps(frozen, indent=2) + "\n"
    )

    prog = {
        "schema_version": 1,
        "kind": "END_TO_END_JAX_V4_3_HARDWARE_ADAPTIVE_DAYTIME_TO_PACKAGE",
        "status": "STAGE_0_COMPLETE",
        "current_stage": "STAGE_1_EXACT_HASH_3A_3B",
        "plan_path": plan_path,
        "plan_sha256": plan_sha,
        "plan_source_path": r"C:\Users\pries\.cursor\plans\v4.3_daytime_to_package_5333a24e.plan.md",
        "authorisation_timestamp": auth_ts,
        "AUTHORIZE_V4_3_DAYTIME_TO_PACKAGE": True,
        "overnight_execution_authorized": False,
        "portal_upload_authorized": False,
        "portal_mutation_authorized": False,
        "phase10_execution_authorized": False,
        "rental_compute_authorized": False,
        "main_merge_authorized": False,
        "auto_push_authorized": False,
        "ladder_env_implementation_hash": ladder["report"]["env_implementation_hash"],
        "smoke_env_implementation_hash": smoke["env_implementation_hash"],
        "current_worktree_env_implementation_hash": env_impl,
        "r_e5_source_status": source_manifest["r_e5_source_status"],
        "source_snapshot": "experiments/manifests/competition_native_jax_v4_3_source_snapshot.json",
        "frozen_baseline": "experiments/manifests/competition_native_jax_v4_2_frozen_baseline.json",
        "clean_benchmark_tps": clean,
        "operational_smoke_tps": operational,
        "restore_threshold_tps": restore,
        "selected_runtime_identity": {
            "git_commit": source_manifest["git_commit"],
            "source_manifest_sha": source_manifest["source_manifest_sha"],
            "env_semantics_hash": env_sem,
            "env_implementation_hash": env_impl,
            "learner_implementation_hash": learn,
            "training_config_sha": training_config_sha,
        },
        "parent_default": "R_E6_PARENT_COMPATIBLE_COLD_RESTART",
        "never_overwrite": [
            "experiments/manifests/competition_native_jax_throughput_ladder_v4.json",
            "experiments/manifests/competition_native_jax_end_to_end_audit.json",
            "experiments/manifests/competition_native_jax_v4_2_*",
        ],
    }
    (ROOT / "experiments/manifests/competition_native_jax_v4_3_programme_state.json").write_text(
        json.dumps(prog, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "plan_sha256": plan_sha,
                "r_e5_source_status": source_manifest["r_e5_source_status"],
                "env_implementation_hash": env_impl,
                "clean_benchmark_tps": clean,
                "operational_smoke_tps": operational,
                "restore_threshold_tps": restore,
                "source_manifest_sha": source_manifest["source_manifest_sha"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
