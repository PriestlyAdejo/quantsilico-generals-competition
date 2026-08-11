"""Stage 2: formal R-E.5 health audit from reconstructed smoke artefacts."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _finite(x: object) -> bool:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def main() -> int:
    smoke = json.loads((ROOT / "experiments/manifests/competition_native_jax_v4_2_smoke_r_e5.json").read_text())
    snap = json.loads((ROOT / "experiments/manifests/competition_native_jax_v4_3_source_snapshot.json").read_text())
    prog = json.loads((ROOT / "experiments/manifests/competition_native_jax_v4_3_programme_state.json").read_text())

    checks: dict[str, object] = {}
    checks["status_completed"] = smoke.get("status") == "COMPLETED"
    checks["transitions_ge_100k"] = int(smoke.get("transitions", 0)) >= 100_000
    checks["updates_positive"] = int(smoke.get("updates", 0)) > 0
    checks["operational_tps_positive"] = float(smoke.get("valid_learning_tps", 0.0)) > 0.0
    checks["lineage_matches_selected"] = (
        smoke.get("env_implementation_hash") == snap.get("env_implementation_hash")
        and smoke.get("learner_implementation_hash") == snap.get("learner_implementation_hash")
        and smoke.get("env_semantics_hash") == snap.get("env_semantics_hash")
    )
    checks["jax_gpu_used"] = bool(smoke.get("jax_gpu_used"))
    metrics = smoke.get("last_metrics") or {}
    checks["metrics_finite"] = all(_finite(metrics.get(k)) for k in ("entropy", "pg", "ratio", "vloss", "loss"))
    checks["ratio_near_one"] = abs(float(metrics.get("ratio", 0.0)) - 1.0) < 0.05
    checks["entropy_positive"] = float(metrics.get("entropy", 0.0)) > 0.0
    checks["checkpoints_recorded"] = bool(smoke.get("checkpoint_raw")) and bool(smoke.get("checkpoint_ema"))
    checks["no_optimiser_state_expected"] = True  # smoke is params/EMA only by design

    system_ok = all(
        bool(checks[k])
        for k in (
            "status_completed",
            "transitions_ge_100k",
            "updates_positive",
            "operational_tps_positive",
            "lineage_matches_selected",
            "jax_gpu_used",
            "metrics_finite",
            "ratio_near_one",
            "entropy_positive",
            "checkpoints_recorded",
        )
    )

    # Short smoke cannot prove competitive strength; treat learning as inconclusive if system healthy.
    learning_signal = "INCONCLUSIVE_SHORT_SMOKE"
    if not system_ok:
        disposition = "R_E5_UNHEALTHY"
        gate = "BLOCKED_R_E5_HEALTH"
        proceed = False
    else:
        disposition = "R_E5_LEARNING_SIGNAL_INCONCLUSIVE_BUT_SYSTEM_HEALTHY"
        gate = "R_E5_HEALTH_PASS"
        proceed = True

    report = {
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_V4_2_R_E5_HEALTH_GATE",
        "disposition": disposition,
        "gate": gate,
        "proceed": proceed,
        "learning_signal": learning_signal,
        "checks": checks,
        "smoke_manifest": "experiments/manifests/competition_native_jax_v4_2_smoke_r_e5.json",
        "operational_smoke_tps": float(smoke["valid_learning_tps"]),
        "selected_env_implementation_hash": snap["env_implementation_hash"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    out = ROOT / "experiments/manifests/competition_native_jax_v4_2_r_e5_health_gate.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    (ROOT / "experiments/reports/competition_native_jax_v4_2_r_e5_health_gate.md").write_text(
        "\n".join(
            [
                "# R-E.5 health gate",
                "",
                f"**Disposition: `{disposition}`**",
                f"Gate: `{gate}`",
                f"Proceed: `{proceed}`",
                f"Operational TPS: {smoke['valid_learning_tps']}",
                "",
                "## Checks",
                "",
                *[f"- `{k}`: {v}" for k, v in checks.items()],
                "",
            ]
        )
    )

    if proceed:
        prog["status"] = "STAGE_2_COMPLETE"
        prog["current_stage"] = "STAGE_3_V4_3A"
        prog["r_e5_health"] = str(out.relative_to(ROOT)).replace("\\", "/")
    else:
        prog["status"] = "BLOCKED_R_E5_HEALTH"
        prog["current_stage"] = "STAGE_2_FAILED"
    (ROOT / "experiments/manifests/competition_native_jax_v4_3_programme_state.json").write_text(
        json.dumps(prog, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0 if proceed else 1


if __name__ == "__main__":
    raise SystemExit(main())
