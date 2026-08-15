"""Bounded start loop for generals_competition (A100 host was GPU-full).

Retries pod start every RETRY_INTERVAL_S for up to MAX_TRIES; every attempt
is appended to the billing log. Exit codes: 0 = RUNNING, 3 = still blocked
(host full) after the window, 1 = other error. Never deletes or recreates
the pod; the A40 fallback decision is made by the orchestrator, not here.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "dev"))

from runpod_account_probe import load_key, rest  # noqa: E402
from runpod_pod_control import BILLING_LOG, record_billing  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

DEFAULT_POD_ID = "wvjrnxbpcjnr8h"
MAX_TRIES = 24
RETRY_INTERVAL_S = 300
# RUNPOD-ZERO-IDLE-BURN-2026-08-15 section 8: once a fallback resource owns
# the workload, touch this marker to cancel outstanding preferred-resource
# retries so both can never end up RUNNING for one single-GPU workload.
CANCEL_MARKER = REPO / "var/marathon_takeover/stop_pod_retries"


def attempt_start(key: str, pod_id: str) -> tuple[str, str]:
    try:
        pod = rest(key, f"/pods/{pod_id}/start", method="POST")
    except RuntimeError as exc:
        message = str(exc)
        if "not enough free GPUs" in message:
            return "HOST_FULL", message
        return "ERROR", message
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        return "TRANSIENT", str(exc)
    if isinstance(pod, dict) and pod.get("id"):
        record_billing("POD_START", pod)
        return "STARTED", json.dumps(pod, sort_keys=True, default=str)
    return "ERROR", f"unexpected response: {pod}"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", default=DEFAULT_POD_ID)
    parser.add_argument("--max-tries", type=int, default=MAX_TRIES)
    parser.add_argument("--interval", type=int, default=RETRY_INTERVAL_S)
    args = parser.parse_args()
    pod_id = args.pod_id
    key = load_key()
    for index in range(1, args.max_tries + 1):
        if CANCEL_MARKER.exists():
            print(
                f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                f"CANCELLED by marker {CANCEL_MARKER.name} (fallback owns workload)"
            )
            return 4
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        outcome, detail = attempt_start(key, pod_id)
        print(f"{stamp} try {index}/{args.max_tries}: {outcome} {detail[:160]}", flush=True)
        with BILLING_LOG.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "recorded_at_utc": stamp,
                        "provider": "runpod",
                        "action": "POD_START_ATTEMPT",
                        "pod_id": pod_id,
                        "outcome": outcome,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        if outcome == "STARTED":
            return 0
        if outcome == "ERROR":
            return 1
        time.sleep(args.interval)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
