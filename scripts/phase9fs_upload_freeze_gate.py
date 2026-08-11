"""Emit SUBMISSION_UPLOAD_FREEZE_GATE artefact (requires user upload confirmation)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--user-confirmed-upload",
        action="store_true",
        help="Required: operator confirms the ZIP was uploaded to the portal.",
    )
    parser.add_argument("--upload-timestamp", default="", help="ISO8601 upload time from operator.")
    parser.add_argument("--public-version", default="QS-PUBLIC-V001")
    args = parser.parse_args()
    now = datetime.now(timezone.utc).isoformat()

    rec = json.loads(
        (REPO / "experiments" / "manifests" / "phase9fs_first_recommendation_gate.json").read_text(
            encoding="utf-8"
        )
    )
    a = rec["candidate_a"]

    if not args.user_confirmed_upload:
        doc = {
            "schema_version": 1,
            "kind": "SUBMISSION_UPLOAD_FREEZE_GATE",
            "created_at": now,
            "gate_status": "WAITING_FOR_USER_UPLOAD_CONFIRMATION",
            "public_version": args.public_version,
            "frozen": None,
            "monitoring_open": False,
            "note": "Run again with --user-confirmed-upload after manual portal upload.",
            "candidate_snapshot": a,
        }
        status = "WAITING"
    else:
        frozen = {
            "local_version_id": args.public_version,
            "baseline_type": rec.get("baseline_type", "HEURISTIC"),
            "candidate_id": a.get("candidate_id"),
            "stable_candidate_id": a.get("stable_candidate_id"),
            "package_id": a.get("package_id"),
            "source_commit": a.get("source_commit"),
            "zip_sha256": a.get("zip_sha256"),
            "model_sha256": None,
            "configuration_hash": a.get("zip_sha256"),
            "fallback_hash": None,
            "control_mode": "OFF",
            "user_upload_timestamp": args.upload_timestamp or now,
            "frozen_at": now,
        }
        doc = {
            "schema_version": 1,
            "kind": "SUBMISSION_UPLOAD_FREEZE_GATE",
            "created_at": now,
            "gate_status": "PASS",
            "public_version": args.public_version,
            "frozen": frozen,
            "monitoring_open": True,
            "immutable": True,
            "next_gate": "PUBLIC_VERSION_EPOCH_GATE",
            "note": "Upload freeze is immutable; later uploads create new epochs.",
        }
        status = "PASS"

    out = REPO / "experiments" / "manifests" / "phase9fs_submission_upload_freeze_gate.json"
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    (REPO / "experiments" / "reports" / "phase9fs_submission_upload_freeze.md").write_text(
        "\n".join(
            [
                "# SUBMISSION_UPLOAD_FREEZE_GATE",
                "",
                f"Created: {now}",
                "",
                f"- Status: **{doc['gate_status']}**",
                f"- Monitoring open: {doc.get('monitoring_open')}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"gate_status": doc["gate_status"], "status": status}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
