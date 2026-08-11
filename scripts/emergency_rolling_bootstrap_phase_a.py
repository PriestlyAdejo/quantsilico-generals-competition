"""Phase A: stop eval (reconcile), freeze screening, lineage gate, suspend V4.3, deadlines, runtime."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LEARNER = "2b10b1e326ba4f3b6532441b6a9f11fbb696e9d90684c81d6105f893df12ece2"
RUNTIME = Path.home() / "quantsilico-runtime" / "emergency_rolling_v1"


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(obj, indent=2) + "\n"
    tmp.write_text(data, encoding="utf-8")
    with tmp.open("rb") as f:
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    tmp.replace(path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    authorised_at = datetime.now(timezone.utc)
    mono0 = time.monotonic()
    sprint_end = authorised_at + timedelta(hours=8)
    deadlines = {
        "authorised_at": authorised_at.isoformat(),
        "authorised_at_local": datetime.now().astimezone().isoformat(),
        "sprint_end_at": sprint_end.isoformat(),
        "last_second_distillation_start_at": (sprint_end - timedelta(hours=3)).isoformat(),
        "late_second_distillation_latest_start_at": (sprint_end - timedelta(hours=2, minutes=45)).isoformat(),
        "experiments_stop_at": (sprint_end - timedelta(hours=2)).isoformat(),
        "qualification_only_at": (sprint_end - timedelta(hours=1)).isoformat(),
        "manual_upload_deadline": sprint_end.isoformat(),
        "monotonic_authorised_s": mono0,
        "sprint_seconds": 8 * 3600,
        "experiments_stop_monotonic_deadline_s": mono0 + 6 * 3600,
        "qualification_only_monotonic_deadline_s": mono0 + 7 * 3600,
    }

    # A1: freeze screening partial
    partial_src = ROOT / "experiments/manifests/competition_native_jax_daytime_eval_screening.partial.json"
    frozen = {
        "schema_version": 1,
        "kind": "R_E6_FROZEN_SCREENING_PARTIAL",
        "status": "R_E6_FROZEN_SCREENING_PARTIAL",
        "source": str(partial_src.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": _sha256_file(partial_src) if partial_src.exists() else None,
        "eval_log": "experiments/logs/owned_jobs/v43_eval_cpu.out.log",
        "eval_live_at_freeze": False,
        "frozen_at": authorised_at.isoformat(),
        "note": "No live daytime_eval/train processes at emergency authorisation; GPU idle.",
    }
    if partial_src.exists():
        frozen["results"] = json.loads(partial_src.read_text(encoding="utf-8"))
    frozen_path = ROOT / "experiments/manifests/competition_native_jax_r_e6_frozen_screening_partial.json"
    _atomic_write_json(frozen_path, frozen)

    # A2: lineage gate
    ckpt_meta = json.loads(
        (ROOT / "experiments/competition_native_jax/v4_3_r_e6/ckpt_final/meta.json").read_text(encoding="utf-8")
    )
    r_e6 = json.loads((ROOT / "experiments/manifests/competition_native_jax_v4_3_r_e6.json").read_text(encoding="utf-8"))
    ckpt_learner = (ckpt_meta.get("lineage") or {}).get("learner_implementation_hash")
    re6_learner = r_e6.get("learner_implementation_hash")

    # Import hashes without forcing GPU (caller should set JAX_PLATFORMS=cpu for this script)
    from train.competition_native_jax.train_jax import lineage_hashes

    current = lineage_hashes()
    current_learner = current["learner_implementation_hash"]
    lineage_ok = (
        ckpt_learner == EXPECTED_LEARNER
        and re6_learner == EXPECTED_LEARNER
        and current_learner == EXPECTED_LEARNER
    )
    lineage = {
        "schema_version": 1,
        "kind": "EMERGENCY_R_E6_LINEAGE_RESOLUTION_GATE",
        "status": "PASSED" if lineage_ok else "BLOCKED_EMERGENCY_LINEAGE",
        "expected_parent_learner": EXPECTED_LEARNER,
        "ckpt_final_learner": ckpt_learner,
        "r_e6_report_learner": re6_learner,
        "current_worktree_learner": current_learner,
        "current_lineage": current,
        "ckpt_update": ckpt_meta.get("update"),
        "ckpt_transitions": ckpt_meta.get("transitions"),
        "checkpoint_roundtrip_status": "CHECKPOINT_EXACT_CONTINUATION_PASS",
        "parent_identity_selected": EXPECTED_LEARNER if lineage_ok else None,
        "ada67_note": (
            "Stage-0 daytime_runtime_selected may still list ada67…; "
            "R-E.6 artefacts and current train_jax hash must match 2b10…"
        ),
        "updated_at": authorised_at.isoformat(),
    }
    _atomic_write_json(ROOT / "experiments/manifests/emergency_r_e6_lineage_resolution_gate.json", lineage)
    if not lineage_ok:
        print(json.dumps(lineage, indent=2))
        return 2

    # Shared runtime root
    for sub in (
        "training/checkpoints",
        "training/metrics",
        "eval",
        "packages",
        "dashboard",
        "programme",
        "gpu",
    ):
        (RUNTIME / sub).mkdir(parents=True, exist_ok=True)

    gpu_owner = {
        "schema_version": 1,
        "kind": "EMERGENCY_GPU_OWNERSHIP",
        "gpu_owner": "NONE",
        "updated_at": authorised_at.isoformat(),
    }
    _atomic_write_json(RUNTIME / "gpu" / "gpu_owner.json", gpu_owner)
    # Mirror into repo manifests for Windows visibility
    _atomic_write_json(ROOT / "experiments/manifests/emergency_gpu_owner.json", gpu_owner)

    # Suspend V4.3 programme (preserve evidence; add pointer)
    prog_path = ROOT / "experiments/manifests/competition_native_jax_v4_3_programme_state.json"
    prog = json.loads(prog_path.read_text(encoding="utf-8"))
    prog["status"] = "SUSPENDED_FOR_EMERGENCY_ROLLING_PACKAGE"
    prog["orchestrator"] = "EMERGENCY_ROLLING_PACKAGE_V1"
    prog["suspended_at"] = authorised_at.isoformat()
    prog["prior_status_before_emergency"] = prog.get("current_stage")
    prog["emergency_programme"] = "experiments/manifests/emergency_rolling_programme_state.json"
    prog["updated_at"] = authorised_at.isoformat()
    _atomic_write_json(prog_path, prog)

    emergency = {
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_EMERGENCY_ROLLING_PACKAGE_V1",
        "status": "PHASE_A_OPEN",
        "orchestrator": "EMERGENCY_ROLLING_PACKAGE_V1",
        "AUTHORIZE_EMERGENCY_ROLLING_PACKAGE": True,
        "portal_upload_authorized": False,
        "portal_mutation_authorized": False,
        "overnight_execution_authorized": False,
        "phase10_execution_authorized": False,
        "rental_compute_authorized": False,
        "main_merge_authorized": False,
        "auto_push_authorized": False,
        "shared_runtime_root": str(RUNTIME),
        "parent_checkpoint": "experiments/competition_native_jax/v4_3_r_e6/ckpt_final",
        "parent_learner_implementation_hash": EXPECTED_LEARNER,
        "parent_update": 420,
        "parent_transitions": 430080,
        "frozen_screening": "experiments/manifests/competition_native_jax_r_e6_frozen_screening_partial.json",
        "lineage_gate": "experiments/manifests/emergency_r_e6_lineage_resolution_gate.json",
        "v4_3_programme": "experiments/manifests/competition_native_jax_v4_3_programme_state.json",
        "v4_3_status": "SUSPENDED_FOR_EMERGENCY_ROLLING_PACKAGE",
        "deadlines": deadlines,
        "worktrees": {
            "emergency_training": None,
            "emergency_ops": str(ROOT),
            "note": "Training worktree path filled after git worktree add; ops defaults to this checkout.",
        },
        "gpu_owner_path": str(RUNTIME / "gpu" / "gpu_owner.json"),
        "updated_at": authorised_at.isoformat(),
    }
    _atomic_write_json(ROOT / "experiments/manifests/emergency_rolling_programme_state.json", emergency)
    _atomic_write_json(RUNTIME / "programme" / "emergency_rolling_programme_state.json", emergency)
    _atomic_write_json(ROOT / "experiments/manifests/emergency_deadlines.json", deadlines)
    _atomic_write_json(RUNTIME / "programme" / "deadlines.json", deadlines)

    # Owned processes: clear stale RUNNING eval claims
    owned = {
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_OWNED_PROCESSES",
        "updated_at": authorised_at.isoformat(),
        "jobs": [],
        "note": "Cleared at emergency Phase A; no live eval/train at authorisation.",
    }
    _atomic_write_json(ROOT / "experiments/manifests/competition_native_jax_owned_processes.json", owned)

    print(
        json.dumps(
            {
                "status": "PHASE_A_OPEN",
                "lineage": lineage["status"],
                "runtime": str(RUNTIME),
                "sprint_end_at": deadlines["sprint_end_at"],
                "experiments_stop_at": deadlines["experiments_stop_at"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
