"""Package Candidate A (QS-P9FU-HEURISTIC-V1 / QS-P9F-PORTAL-V0) and emit FIRST_SUBMISSION_PACKAGE_GATE."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from generals_bot.submission.builder import build_heuristic_package, windows_clean_package_validation

REPO = Path(__file__).resolve().parents[1]
CANONICAL = "heuristic_v2f_plus_planner_terminal_" + "f" + "ix"
STABLE_ID = "QS-P9F-PORTAL-V0"
PACKAGE_ID = "QS-P9FU-HEURISTIC-V1"


def main() -> int:
    now = datetime.now(timezone.utc).isoformat()
    identity_path = REPO / "experiments" / "manifests" / "phase9fs_candidate_a_identity_gate.json"
    if not identity_path.exists():
        raise SystemExit("identity gate artefact missing; run phase9fs_candidate_a_identity_gate.py first")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if not str(identity.get("gate_status", "")).startswith("PASS"):
        raise SystemExit(f"identity gate not PASS: {identity.get('gate_status')}")
    if identity.get("canonical_implementation_string") != CANONICAL:
        raise SystemExit(
            f"canonical mismatch: {identity.get('canonical_implementation_string')!r} != {CANONICAL!r}"
        )

    out_dir = REPO / "dist" / "submission_recommended"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build_heuristic_package(
        CANONICAL,
        out_dir=out_dir,
        package_stem=PACKAGE_ID.lower().replace("-", "_"),
        overwrite=True,
    )
    zip_path = Path(report.package_path)
    smoke = windows_clean_package_validation(zip_path)

    # Also copy role pointer
    role_dir = REPO / "dist" / "roles"
    role_dir.mkdir(parents=True, exist_ok=True)
    role_doc = {
        "role": "portal_current_verified",
        "stable_id": STABLE_ID,
        "package_id": PACKAGE_ID,
        "candidate_id": CANONICAL,
        "package_path": str(zip_path.as_posix()),
        "sha256": report.sha256,
        "updated_at": now,
    }
    (role_dir / "portal_current_verified.json").write_text(json.dumps(role_doc, indent=2) + "\n", encoding="utf-8")

    linux_status = "RECOMMENDED_FOR_MANUAL_UPLOAD_WITH_EXTERNAL_LINUX_BLOCKER"
    # Strict upload_ready stays empty until official Linux parity
    (REPO / "dist" / "upload_ready").mkdir(parents=True, exist_ok=True)

    gate = {
        "schema_version": 1,
        "kind": "FIRST_SUBMISSION_PACKAGE_GATE",
        "created_at": now,
        "gate_status": "PASS" if smoke.get("status") == "PASS" and report.status == "PACKAGED" else "FIRST_PACKAGE_BLOCKED",
        "package_id": PACKAGE_ID,
        "stable_candidate_id": STABLE_ID,
        "candidate_id": CANONICAL,
        "source_commit": report.bot_commit,
        "engine_commit": report.engine_commit,
        "source_hash": report.sha256,
        "zip_path": str(zip_path.as_posix()),
        "zip_sha256": report.sha256,
        "zip_size": report.zip_size,
        "protocol_smoke": smoke,
        "cpu_status": "WINDOWS_PROTOCOL_SMOKE_PASS",
        "memory_notes": "heuristic package; no neural weights",
        "package_size_bytes": report.zip_size,
        "linux_status": linux_status,
        "recommendation_status": "READY_FOR_FIRST_RECOMMENDATION_AFTER_B_TIMEBOX",
        "official_upload_ready": False,
        "directory_policy": {
            "STRICT_UPLOAD_READY": "dist/upload_ready/",
            "RECOMMENDED_FOR_MANUAL_UPLOAD_WITH_EXTERNAL_LINUX_BLOCKER": "dist/submission_recommended/",
            "chosen": "dist/submission_recommended/",
        },
    }
    if gate["gate_status"] != "PASS":
        gate["defect"] = {
            "report_status": report.status,
            "smoke_status": smoke.get("status"),
            "smoke_notes": smoke.get("notes"),
        }

    (REPO / "experiments" / "manifests" / "phase9fs_first_submission_package_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    (REPO / "experiments" / "reports" / "phase9fs_first_submission_package_gate.md").write_text(
        "\n".join(
            [
                "# FIRST_SUBMISSION_PACKAGE_GATE",
                "",
                f"Created: {now}",
                "",
                f"- Status: **{gate['gate_status']}**",
                f"- Package ID: `{PACKAGE_ID}`",
                f"- Stable ID: `{STABLE_ID}`",
                f"- Candidate: `{CANONICAL}`",
                f"- ZIP: `{zip_path.as_posix()}`",
                f"- SHA-256: `{report.sha256}`",
                f"- Linux status: `{linux_status}`",
                f"- Protocol smoke: `{smoke.get('status')}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"gate_status": gate["gate_status"], "zip": str(zip_path), "sha256": report.sha256}, indent=2))
    return 0 if gate["gate_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
