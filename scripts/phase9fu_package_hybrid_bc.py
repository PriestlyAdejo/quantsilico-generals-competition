"""Package QS-P9FU-HYBRID-BC-V1 from the phase9f CNN BC checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

from generals_bot.submission.builder import build_hybrid_bc_package

REPO = Path(__file__).resolve().parents[1]
BC_JSON = REPO / "experiments" / "phase9f_cnn_ranker_v1" / "checkpoints" / "bc" / "model.json"
CANDIDATE_ID = "QS-P9FU-HYBRID-BC-V1"


def main() -> int:
    if not BC_JSON.is_file():
        raise SystemExit(f"BC checkpoint missing: {BC_JSON}")
    report = build_hybrid_bc_package(
        BC_JSON,
        candidate_id=CANDIDATE_ID,
        fallback_policy_name="heuristic_v2f_plus_planner_terminal_fix",
        promote=True,
    )
    out = {
        "candidate_id": CANDIDATE_ID,
        "status": report.status,
        "package_path": report.package_path,
        "sha256": report.sha256,
        "build_hash": (report.extra or {}).get("build_hash"),
        "zip_size": report.zip_size,
        "notes": report.notes,
    }
    print(json.dumps(out, indent=2))
    return 0 if report.status == "PACKAGED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
