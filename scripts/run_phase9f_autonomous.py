"""Phase 9F autonomous controller stub — resume after foundation packaging milestone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATE = REPO / "experiments/manifests/phase9f_autonomous_run_state.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--deadline", default="2026-08-04T22:00:00+01:00")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--concurrency", default="auto")
    args = p.parse_args()
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}
    print(
        json.dumps(
            {
                "status": "RESUME_POINT",
                "stage": state.get("stage"),
                "gates": state.get("gates"),
                "next": [
                    "persistent env workers across PPO chunks",
                    "structured fog memory on learned/hybrid path",
                    "specialist teacher gates",
                    "teacher dataset + BC smoke",
                    "replace portal best_overall only when a stronger CPU-qualified candidate exists",
                ],
                "deadline": args.deadline,
                "device": args.device,
                "packages": "dist/upload_ready/",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
