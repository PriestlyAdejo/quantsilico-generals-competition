"""RunPod pod lifecycle control for the QuantSilico Marathon.

Official REST endpoints extracted from runpodctl commit 6fc6f8b5. Every paid
state change is appended to var/marathon_takeover/runpod_billing_log.jsonl
(billing-evidence requirement, RUNPOD-SPEND-2026-08-15). Subcommands:
status | start | stop | poll-ready | exec-list <pod-dir>. Key is read from
~/.runpod/config.toml and never printed.

Usage:
  python scripts/dev/runpod_pod_control.py status
  python scripts/dev/runpod_pod_control.py start wvjrnxbpcjnr8h
  python scripts/dev/runpod_pod_control.py poll-ready wvjrnxbpcjnr8h --timeout 600
  python scripts/dev/runpod_pod_control.py stop wvjrnxbpcjnr8h
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "dev"))

from runpod_account_probe import gql, load_key, rest  # noqa: E402

BILLING_LOG = REPO / "var/marathon_takeover/runpod_billing_log.jsonl"


def record_billing(action: str, pod: dict, extra: dict | None = None) -> None:
    BILLING_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provider": "runpod",
        "action": action,
        "pod_id": pod.get("id"),
        "pod_name": pod.get("name"),
        "desired_status": pod.get("desiredStatus"),
        "gpu_count": pod.get("gpuCount"),
        "machine_id": pod.get("machineId"),
        "cost_per_hr_at_action": pod.get("costPerHr"),
    }
    if extra:
        entry.update(extra)
    with BILLING_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def account_snapshot(key: str) -> dict:
    me = gql(
        key,
        """query { myself { clientBalance currentSpendPerHr spendLimit } }""",
    )["myself"]
    return me


def cmd_status(key: str) -> int:
    pods = rest(key, "/pods")
    account = account_snapshot(key)
    print(
        json.dumps(
            {"account": account, "pods": pods}, indent=2, sort_keys=True, default=str
        )
    )
    return 0


def _balance_context(before: dict) -> dict:
    return {
        "balance_before": before["clientBalance"],
        "spend_per_hr_before": before["currentSpendPerHr"],
    }


def cmd_start(key: str, pod_id: str) -> int:
    before = account_snapshot(key)
    pod = rest(key, f"/pods/{pod_id}/start", method="POST")
    if not isinstance(pod, dict) or "id" not in pod:
        print(f"unexpected start response: {pod}", file=sys.stderr)
        return 1
    record_billing("POD_START", pod, _balance_context(before))
    print(json.dumps({"started": pod}, indent=2, sort_keys=True, default=str))
    return 0


def cmd_stop(key: str, pod_id: str) -> int:
    before = account_snapshot(key)
    pod = rest(key, f"/pods/{pod_id}/stop", method="POST")
    if not isinstance(pod, dict) or "id" not in pod:
        print(f"unexpected stop response: {pod}", file=sys.stderr)
        return 1
    record_billing("POD_STOP", pod, _balance_context(before))
    print(json.dumps({"stopped": pod}, indent=2, sort_keys=True, default=str))
    return 0


def cmd_poll_ready(key: str, pod_id: str, timeout: int, interval: int) -> int:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        pod = rest(key, f"/pods/{pod_id}")
        runtime = pod.get("runtime", {}) if isinstance(pod, dict) else {}
        status = runtime.get("desiredStatus")
        actual = None
        if isinstance(pod, dict):
            actual = pod.get("runtime", {}).get("actualStatus") or pod.get("desiredStatus")
        summary = {
            "desired": status,
            "actual": actual,
            "uptime_seconds": runtime.get("uptimeSeconds"),
        }
        if summary != last:
            print(json.dumps(summary, sort_keys=True, default=str), flush=True)
            last = summary
        if actual == "RUNNING":
            record_billing("POD_RUNNING_OBSERVED", pod if isinstance(pod, dict) else {"id": pod_id})
            print(json.dumps(pod, indent=2, sort_keys=True, default=str))
            return 0
        time.sleep(interval)
    print(f"pod not RUNNING after {timeout}s", file=sys.stderr)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    for name in ("start", "stop", "poll-ready"):
        p = sub.add_parser(name)
        p.add_argument("pod_id")
        if name == "poll-ready":
            p.add_argument("--timeout", type=int, default=900)
            p.add_argument("--interval", type=int, default=15)
    args = parser.parse_args()
    key = load_key()
    if args.command == "status":
        return cmd_status(key)
    if args.command == "start":
        return cmd_start(key, args.pod_id)
    if args.command == "stop":
        return cmd_stop(key, args.pod_id)
    if args.command == "poll-ready":
        return cmd_poll_ready(key, args.pod_id, args.timeout, args.interval)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
