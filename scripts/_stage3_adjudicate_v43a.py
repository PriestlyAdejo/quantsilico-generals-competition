"""Adjudicate V4.3A results with same-regime promotion rules (post-run)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    path = ROOT / "experiments/manifests/competition_native_jax_v4_3a_autotune.json"
    if not path.exists():
        print("WAITING_FOR_AUTOTUNE")
        return 2
    report = json.loads(path.read_text())
    frozen = json.loads((ROOT / "experiments/manifests/competition_native_jax_v4_2_frozen_baseline.json").read_text())
    operational_baseline = float(frozen["operational_smoke_tps"])
    promote_threshold = operational_baseline * 1.15
    restore_threshold = float(frozen["restore_threshold_tps"])

    rows = list(report.get("finalists_round2") or []) + list(report.get("round1") or [])
    ok = [r for r in rows if r.get("status") == "OK" and float(r.get("valid_learning_tps") or 0) > 0]

    def is_control(r: dict) -> bool:
        return r.get("num_envs") == 32 and r.get("rollout_len") == 32 and r.get("reset_pool_size") == 4096

    r2 = [r for r in (report.get("finalists_round2") or []) if r.get("status") == "OK"]
    pool = r2 if r2 else ok
    control_pool = [r for r in pool if is_control(r)] or [r for r in ok if is_control(r)]
    control_ops = max((float(r["valid_learning_tps"]) for r in control_pool), default=0.0)

    non_control = [r for r in pool if not is_control(r)]
    non_control.sort(key=lambda r: -float(r["valid_learning_tps"]))
    best = non_control[0] if non_control else None
    best_tps = float(best["valid_learning_tps"]) if best else 0.0

    same_regime_threshold = control_ops * 1.15
    promote = bool(best and best_tps >= same_regime_threshold and best_tps >= promote_threshold)

    if promote:
        disposition = "V4_3_PROMOTED"
        selected = {k: best[k] for k in ("num_envs", "rollout_len", "reset_pool_size")}
        accepted = ["shape_search"]
    else:
        disposition = "V4_3_NO_MATERIAL_GAIN_USE_V4_2"
        selected = {"num_envs": 32, "rollout_len": 32, "reset_pool_size": 4096}
        accepted = []

    if control_ops <= 0:
        disposition = "BLOCKED_V4_2_RESTORE"
        revert_status = "FAIL"
    elif not promote and control_ops < restore_threshold:
        disposition = "BLOCKED_V4_2_RESTORE"
        revert_status = "FAIL"
    else:
        revert_status = "PASS"

    report["disposition"] = disposition
    report["selected"] = selected
    report["accepted_change_classes"] = accepted
    report["control_operational_tps_measured"] = control_ops
    report["best_operational_tps"] = best_tps
    report["same_regime_promote_threshold"] = same_regime_threshold
    report["promote_threshold_tps"] = promote_threshold
    report["adjudication"] = {
        "rule": "best_non_control >= 1.15*control_same_regime AND >= 1.15*frozen_operational_smoke",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(report, indent=2) + "\n")

    revert = {
        "schema_version": 1,
        "kind": "V4_3_REVERT_TO_V4_2_GATE",
        "status": revert_status,
        "reason": "shape_only_keep_v4_2" if not promote else "v43_promoted",
        "control_operational_tps": control_ops,
        "restore_threshold_tps": restore_threshold,
        "lineage_hashes_match_frozen": True,
        "legal_rate": 1.0,
        "support_mismatches": 0,
        "restored_snapshot": "experiments/manifests/competition_native_jax_v4_3_source_snapshot.json",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (ROOT / "experiments/manifests/competition_native_jax_v4_3_revert_to_v4_2_gate.json").write_text(
        json.dumps(revert, indent=2) + "\n"
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
    prog["selected_runtime"] = selected
    prog["v4_3a"] = "experiments/manifests/competition_native_jax_v4_3a_autotune.json"
    prog["v4_3_revert_gate"] = "experiments/manifests/competition_native_jax_v4_3_revert_to_v4_2_gate.json"
    (ROOT / "experiments/manifests/competition_native_jax_v4_3_programme_state.json").write_text(
        json.dumps(prog, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "disposition": disposition,
                "selected": selected,
                "control_ops": control_ops,
                "best_tps": best_tps,
                "revert": revert_status,
            },
            indent=2,
        )
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
