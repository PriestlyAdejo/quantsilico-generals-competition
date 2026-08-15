"""Marathon execution lease: mutual exclusion for recurring continuation loops.

Amendment ELITE-REPLAY-INTELLIGENCE-DURABLE-CONTINUATION-2026-08-15 §34.

Prevents duplicate loop iterations from provisioning pods, launching the same
NEXT_SAFE_ACTION, adjudicating twice, or writing ACTIVE_STATE concurrently.
The lease is a small JSON file with owner identity, acquisition timestamp,
heartbeat, and a stale timeout so a crashed owner can NEVER permanently block
the Marathon: once stale, the next iteration may take over (logged, never
silently).

Usage:
    python scripts/dev/marathon_execution_lease.py acquire --owner qoder-quest-979999d1
    python scripts/dev/marathon_execution_lease.py heartbeat
    python scripts/dev/marathon_execution_lease.py status
    python scripts/dev/marathon_execution_lease.py release

Exit codes: 0 ok/acquired; 3 held by another live owner; 4 stale takeover performed.
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import socket
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEASE_PATH = REPO / "experiments" / "marathon" / "execution_lease.json"
STALE_AFTER_S = 90 * 60  # 90 min without heartbeat = stale


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read() -> dict | None:
    try:
        return json.loads(LEASE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def _write(lease: dict) -> None:
    LEASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEASE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(lease, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, LEASE_PATH)


def _age_seconds(lease: dict) -> float:
    try:
        hb = time.strptime(lease["heartbeat_utc"], "%Y-%m-%dT%H:%M:%SZ")
    except (KeyError, ValueError):
        return float("inf")
    return time.time() - calendar.timegm(hb)


def acquire(owner: str) -> int:
    lease = _read()
    if lease is not None and lease.get("held"):
        if _age_seconds(lease) < STALE_AFTER_S and lease.get("owner") != owner:
            print(
                json.dumps(
                    {
                        "acquired": False,
                        "reason": "HELD_BY_LIVE_OWNER",
                        "owner": lease.get("owner"),
                        "heartbeat_utc": lease.get("heartbeat_utc"),
                    }
                )
            )
            return 3
        takeover = lease.get("owner") != owner
        lease = {
            "held": True,
            "owner": owner,
            "acquired_at_utc": _now(),
            "heartbeat_utc": _now(),
            "host": socket.gethostname(),
            "stale_takeover_from": lease.get("owner") if takeover else None,
            "stale_after_s": STALE_AFTER_S,
        }
        _write(lease)
        print(json.dumps({"acquired": True, "stale_takeover": takeover, "lease": lease}))
        return 4 if takeover else 0
    lease = {
        "held": True,
        "owner": owner,
        "acquired_at_utc": _now(),
        "heartbeat_utc": _now(),
        "host": socket.gethostname(),
        "stale_takeover_from": None,
        "stale_after_s": STALE_AFTER_S,
    }
    _write(lease)
    print(json.dumps({"acquired": True, "stale_takeover": False, "lease": lease}))
    return 0


def heartbeat() -> int:
    lease = _read()
    if lease is None or not lease.get("held"):
        print(json.dumps({"ok": False, "reason": "NO_LEASE"}))
        return 3
    lease["heartbeat_utc"] = _now()
    _write(lease)
    print(
        json.dumps(
            {"ok": True, "owner": lease.get("owner"), "heartbeat_utc": lease["heartbeat_utc"]}
        )
    )
    return 0


def status() -> int:
    lease = _read()
    if lease is None or not lease.get("held"):
        print(json.dumps({"held": False}))
        return 0
    age = _age_seconds(lease)
    print(
        json.dumps(
            {
                "held": True,
                "stale": age >= STALE_AFTER_S,
                "age_s": round(age, 1),
                **lease,
            }
        )
    )
    return 0


def release() -> int:
    lease = _read()
    if lease is None or not lease.get("held"):
        print(json.dumps({"released": False, "reason": "NO_LEASE"}))
        return 0
    lease["held"] = False
    lease["released_at_utc"] = _now()
    _write(lease)
    print(json.dumps({"released": True}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["acquire", "heartbeat", "status", "release"])
    parser.add_argument("--owner", default=os.environ.get("MARATHON_LEASE_OWNER", "unknown"))
    args = parser.parse_args()
    if args.action == "acquire":
        return acquire(args.owner)
    if args.action == "heartbeat":
        return heartbeat()
    if args.action == "status":
        return status()
    return release()


if __name__ == "__main__":
    sys.exit(main())
