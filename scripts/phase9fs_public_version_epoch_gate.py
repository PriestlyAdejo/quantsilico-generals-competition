"""PUBLIC_VERSION_EPOCH_GATE — record first public observation after upload freeze."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard-timestamp", default="")
    parser.add_argument("--first-public-game-timestamp", default="")
    parser.add_argument("--public-player-identity", default="QuantSilico")
    parser.add_argument("--public-version-epoch", default="QS-PUBLIC-V001")
    parser.add_argument("--initial-elo", default="")
    parser.add_argument("--initial-rank", default="")
    parser.add_argument("--initial-record", default="")
    args = parser.parse_args()
    now = datetime.now(timezone.utc).isoformat()

    freeze_path = REPO / "experiments" / "manifests" / "phase9fs_submission_upload_freeze_gate.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8")) if freeze_path.exists() else {}
    if freeze.get("gate_status") != "PASS":
        doc = {
            "schema_version": 1,
            "kind": "PUBLIC_VERSION_EPOCH_GATE",
            "created_at": now,
            "gate_status": "BLOCKED_UNTIL_UPLOAD_FREEZE",
            "note": "Upload freeze must PASS before public epoch recording.",
        }
    elif not (args.leaderboard_timestamp or args.first_public_game_timestamp):
        doc = {
            "schema_version": 1,
            "kind": "PUBLIC_VERSION_EPOCH_GATE",
            "created_at": now,
            "gate_status": "WAITING_FOR_PUBLIC_OBSERVATION",
            "upload_freeze": freeze.get("frozen"),
            "monitoring_active": True,
            "note": "Monitoring may begin after upload freeze; epoch completes when public observation arrives.",
        }
    else:
        doc = {
            "schema_version": 1,
            "kind": "PUBLIC_VERSION_EPOCH_GATE",
            "created_at": now,
            "gate_status": "PASS",
            "upload_freeze_immutable": freeze.get("frozen"),
            "epoch": {
                "first_observed_leaderboard_timestamp": args.leaderboard_timestamp or None,
                "first_observed_public_game_timestamp": args.first_public_game_timestamp or None,
                "public_player_identity": args.public_player_identity,
                "public_version_epoch": args.public_version_epoch,
                "initial_elo": args.initial_elo or None,
                "initial_rank": args.initial_rank or None,
                "initial_public_record": args.initial_record or None,
            },
        }

    (REPO / "experiments" / "manifests" / "phase9fs_public_version_epoch_gate.json").write_text(
        json.dumps(doc, indent=2) + "\n", encoding="utf-8"
    )
    (REPO / "experiments" / "reports" / "phase9fs_public_version_epoch.md").write_text(
        f"# PUBLIC_VERSION_EPOCH_GATE\n\nCreated: {now}\n\nStatus: **{doc['gate_status']}**\n",
        encoding="utf-8",
    )
    print(json.dumps({"gate_status": doc["gate_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
