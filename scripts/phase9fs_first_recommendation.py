"""FIRST_RECOMMENDATION_GATE — recommend QS-PUBLIC-V001 without waiting for Candidate C."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    now = datetime.now(timezone.utc).isoformat()
    pkg = json.loads(
        (REPO / "experiments" / "manifests" / "phase9fs_first_submission_package_gate.json").read_text(
            encoding="utf-8"
        )
    )
    identity = json.loads(
        (REPO / "experiments" / "manifests" / "phase9fs_candidate_a_identity_gate.json").read_text(
            encoding="utf-8"
        )
    )
    b_path = REPO / "experiments" / "manifests" / "phase9fs_candidate_b_fast_lane.json"
    b_doc = json.loads(b_path.read_text(encoding="utf-8")) if b_path.exists() else None

    a_ok = pkg.get("gate_status") == "PASS" and str(identity.get("gate_status", "")).startswith("PASS")
    b_class = (b_doc or {}).get("classification", "CANDIDATE_B_NOT_COMPLETED")
    b_ok_for_rec = b_class in {
        "CANDIDATE_B_QUALIFIED",
        "CANDIDATE_B_BLOCKED_SEMANTICS",
        "CANDIDATE_B_BLOCKED_RUNTIME",
        "CANDIDATE_B_BLOCKED_CPU",
        "CANDIDATE_B_NOT_COMPLETED",
    }

    gate_status = "PASS" if a_ok and b_ok_for_rec else "FAIL"
    linux_status = pkg.get("linux_status", "NOT_READY")
    recommended_version = "QS-PUBLIC-V001"
    baseline_type = "HEURISTIC"

    doc = {
        "schema_version": 1,
        "kind": "FIRST_RECOMMENDATION_GATE",
        "created_at": now,
        "gate_status": gate_status,
        "recommended_public_version": recommended_version,
        "baseline_type": baseline_type,
        "candidate_a": {
            "package_id": pkg.get("package_id"),
            "stable_candidate_id": pkg.get("stable_candidate_id"),
            "candidate_id": pkg.get("candidate_id"),
            "zip_path": pkg.get("zip_path"),
            "zip_sha256": pkg.get("zip_sha256"),
            "source_commit": pkg.get("source_commit"),
            "linux_status": linux_status,
        },
        "candidate_b": {
            "classification": b_class,
            "defect": (b_doc or {}).get("defect"),
            "included_in_v001": b_class == "CANDIDATE_B_QUALIFIED",
        },
        "candidate_c": {
            "status": "NOT_REQUIRED_FOR_V001",
            "note": "Do not hold QS-PUBLIC-V001 for Candidate C / PPO repair / throughput.",
        },
        "upload_policy": {
            "agent_uploads": False,
            "user_manual_upload_required": True,
            "directory": "dist/submission_recommended/"
            if linux_status == "RECOMMENDED_FOR_MANUAL_UPLOAD_WITH_EXTERNAL_LINUX_BLOCKER"
            else "dist/upload_ready/",
        },
        "next_gate_after_user_upload": "SUBMISSION_UPLOAD_FREEZE_GATE",
    }

    (REPO / "experiments" / "manifests" / "phase9fs_first_recommendation_gate.json").write_text(
        json.dumps(doc, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# FIRST_RECOMMENDATION_GATE",
        "",
        f"Created: {now}",
        "",
        f"- Gate: **{gate_status}**",
        f"- Recommend manual upload of **{recommended_version}**",
        f"- Baseline type: `{baseline_type}`",
        f"- Candidate A: `{pkg.get('candidate_id')}` / `{pkg.get('package_id')}`",
        f"- ZIP: `{pkg.get('zip_path')}`",
        f"- SHA-256: `{pkg.get('zip_sha256')}`",
        f"- Linux status: `{linux_status}`",
        f"- Candidate B: `{b_class}` (does not block V001)",
        f"- Candidate C: not required for V001",
        "",
        "Agent will **not** upload. After you upload, confirm so SUBMISSION_UPLOAD_FREEZE_GATE can freeze identity.",
        "",
    ]
    (REPO / "experiments" / "reports" / "phase9fs_first_recommendation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    # Also drop a short operator note beside the ZIP
    rec_dir = REPO / "dist" / "submission_recommended"
    rec_dir.mkdir(parents=True, exist_ok=True)
    (rec_dir / "README_MANUAL_UPLOAD.md").write_text(
        "\n".join(
            [
                "# Manual upload recommendation (QS-PUBLIC-V001)",
                "",
                f"- Package: `{pkg.get('zip_path')}`",
                f"- SHA-256: `{pkg.get('zip_sha256')}`",
                f"- Candidate: `{pkg.get('candidate_id')}`",
                f"- Stable ID: `{pkg.get('stable_candidate_id')}`",
                f"- Linux: `{linux_status}` (residual risk accepted by operator)",
                "",
                "Do not auto-upload. After portal upload, tell the agent so upload freeze can open monitoring.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"gate_status": gate_status, "version": recommended_version, "b": b_class}, indent=2))
    return 0 if gate_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
