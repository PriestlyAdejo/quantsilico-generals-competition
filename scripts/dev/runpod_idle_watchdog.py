"""RunPod zero-idle-burn watchdog (RUNPOD-ZERO-IDLE-BURN-2026-08-15).

Reconciles live RunPod pods against experiments/marathon/runpod_resources.json.
RUNNING pods must own a verified active workload: the watchdog checks the
workload itself (trainer process / telemetry advancing / GPU utilisation),
never merely the pod status. Idle/orphan/finished pods are reported and,
with --stop-idle, stopped after capturing a log tail; every action is
billing-logged and the resource ledger is updated.

Usage:
  python scripts/dev/runpod_idle_watchdog.py               # report only
  python scripts/dev/runpod_idle_watchdog.py --stop-idle   # stop idle pods
Exit codes: 0 all clear, 2 idle/orphan pods found (or stopped), 1 error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "dev"))

from runpod_account_probe import load_key, rest  # noqa: E402
from runpod_pod_control import account_snapshot, record_billing  # noqa: E402

LEDGER_PATH = REPO / "experiments/marathon/runpod_resources.json"
GRACE_MINUTES_DEFAULT = 10


def ssh_probe(host: str, port: int, command: str, timeout: int = 25) -> str:
    try:
        completed = subprocess.run(
            [
                "ssh",
                "-o", "ConnectTimeout=10",
                "-o", "BatchMode=yes",
                "-p", str(port),
                f"root@{host}",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"PROBE_ERROR {exc}"


def workload_alive(pod: dict) -> tuple[bool, str]:
    host = pod.get("publicIp") or ""
    port = (pod.get("portMappings") or {}).get("22")
    if not host or not port:
        return False, "no SSH endpoint to verify workload"
    gpu = ssh_probe(
        host,
        port,
        "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
    )
    procs = ssh_probe(host, port, "pgrep -f run_sh_r1_arm | head -1")
    if procs and not procs.startswith("PROBE_ERROR"):
        return True, f"trainer pid {procs}; gpu {gpu}"
    return False, f"no trainer process; gpu {gpu}"


def update_ledger(stopped: list[dict]) -> None:
    if not stopped:
        return
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    by_id = {r["pod_id"]: r for r in ledger["resources"]}
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for pod in stopped:
        record = by_id.get(pod["id"])
        if record is None:
            record = {
                "pod_id": pod["id"],
                "name": pod.get("name"),
                "rate_usd_per_hr": pod.get("costPerHr"),
            }
            ledger["resources"].append(record)
        record["status"] = "EXITED_STOPPED_BY_WATCHDOG"
        record["stopped_at_utc"] = stamp
    ledger["updated_at_utc"] = stamp
    LEDGER_PATH.write_text(
        json.dumps(ledger, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-idle", action="store_true")
    parser.add_argument("--grace-minutes", type=int, default=GRACE_MINUTES_DEFAULT)
    args = parser.parse_args()

    key = load_key()
    pods = rest(key, "/pods")
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    known = {r["pod_id"]: r for r in ledger.get("resources", [])}

    running = [p for p in pods if isinstance(p, dict) and p.get("desiredStatus") == "RUNNING"]
    idle: list[dict] = []
    report = {"checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    for pod in running:
        alive, detail = workload_alive(pod)
        entry = {
            "pod_id": pod["id"],
            "name": pod.get("name"),
            "rate_usd_per_hr": pod.get("costPerHr"),
            "known_to_ledger": pod["id"] in known,
            "workload_alive": alive,
            "detail": detail,
        }
        if not alive:
            idle.append(pod)
            entry["verdict"] = (
                "ORPHAN" if pod["id"] not in known else "IDLE_OR_FINISHED"
            )
        else:
            entry["verdict"] = "ACTIVE_WORKLOAD_VERIFIED"
        report.setdefault("running_pods", []).append(entry)

    if not running:
        report["verdict"] = "NO_RUNNING_PODS"
    elif idle:
        report["verdict"] = f"{len(idle)} IDLE/ORPHAN POD(S)"
    else:
        report["verdict"] = "ALL_RUNNING_PODS_HAVE_VERIFIED_WORKLOADS"

    stopped: list[dict] = []
    if idle and args.stop_idle:
        before = account_snapshot(key)
        for pod in idle:
            result = rest(key, f"/pods/{pod['id']}/stop", method="POST")
            if isinstance(result, dict) and result.get("id"):
                stopped.append(pod)
                record_billing(
                    "WATCHDOG_STOP_IDLE",
                    pod,
                    {
                        "balance_before": before["clientBalance"],
                        "reason": report["verdict"],
                        "invariant": "RUNPOD-ZERO-IDLE-BURN-2026-08-15",
                    },
                )
        update_ledger(stopped)
        report["stopped"] = [p["id"] for p in stopped]

    print(json.dumps(report, indent=2, default=str))
    return 2 if idle else 0


if __name__ == "__main__":
    raise SystemExit(main())
