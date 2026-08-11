"""Package QS-P9FU-HEURISTIC-TACTICAL-V2 under submission/packages/."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from generals_bot.submission.builder import (
    build_heuristic_package,
    promote_package_to_submission,
)

REPO = Path(__file__).resolve().parents[1]
CANDIDATE = "heuristic_v2f_tactical_attack_v2"
PACKAGE_ID = "QS-P9FU-HEURISTIC-TACTICAL-V2"


def main() -> int:
    now = datetime.now(timezone.utc).isoformat()
    staging = Path(tempfile.mkdtemp(prefix="qs_p9fu_tactical_v2_"))
    try:
        report = build_heuristic_package(
            CANDIDATE,
            out_dir=staging,
            package_stem=PACKAGE_ID.lower().replace("-", "_"),
            overwrite=True,
        )
        zip_path = Path(report.package_path)
        promoted = promote_package_to_submission(PACKAGE_ID, zip_path)

        # Correct architecture label on sidecar (promote helper defaults hybrid_bc).
        manifest_path = Path(promoted["package_path"]).parent / "package_manifest.json"
        if manifest_path.is_file():
            man = json.loads(manifest_path.read_text(encoding="utf-8"))
            man["architecture"] = "heuristic"
            man["policy_candidate"] = CANDIDATE
            man["package_id"] = PACKAGE_ID
            man["built_at"] = now
            manifest_path.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")

        summary = {
            "schema_version": 1,
            "kind": "QS_P9FU_HEURISTIC_TACTICAL_V2_PACKAGE",
            "created_at": now,
            "candidate": CANDIDATE,
            "package_id": PACKAGE_ID,
            "build_hash": promoted["build_hash"],
            "sha256": promoted["sha256"],
            "package_path": promoted["relative_path"],
            "bot_commit": report.bot_commit,
            "engine_commit": report.engine_commit,
            "report_status": report.status,
        }
        out_manifest = REPO / "experiments" / "manifests" / "phase9fu_tactical_v2_package.json"
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        out_manifest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0 if report.status == "PACKAGED" else 1
    finally:
        # staging cleaned by leaving temp dir; promote already copied bytes
        pass


if __name__ == "__main__":
    raise SystemExit(main())
