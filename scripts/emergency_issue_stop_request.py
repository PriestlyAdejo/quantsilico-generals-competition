"""Issue cooperative STOP_REQUEST for first-learned distill handoff."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path.home() / "quantsilico-runtime" / "emergency_rolling_v1"
STOP = RUNTIME / "training" / "STOP_REQUEST"
STOP_REPO = ROOT / "experiments/competition_native_jax/emergency_rolling_v1" / "STOP_REQUEST"


def main() -> int:
    payload = {
        "reason": "FIRST_LEARNED_DISTILL_HANDOFF",
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "grace_note": "READY route + teacher + dataset path; COMPLETE boundary cooperative stop",
    }
    STOP.parent.mkdir(parents=True, exist_ok=True)
    STOP_REPO.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    STOP.write_text(text, encoding="utf-8")
    STOP_REPO.write_text(text, encoding="utf-8")
    print("STOP_REQUEST_WRITTEN", STOP, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
