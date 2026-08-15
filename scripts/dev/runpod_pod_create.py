"""Provision the pre-declared A40 fallback pod for SH-R1 GPU arms.

Uses the official GraphQL podFindAndDeployOnDemand mutation (runpodctl
api/pod.go, commit 6fc6f8b5) which finds a suitable machine itself. Tries
each HIGH-stock data center in order. Refuses duplicates. Billing-logged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "dev"))

from runpod_account_probe import GQL_API, _http, load_key, rest  # noqa: E402
from runpod_pod_control import account_snapshot, record_billing  # noqa: E402

DEFAULT_POD_NAME = "quantisilico_sh_r1_a40"
PUBLIC_KEY_PATH = Path.home() / ".ssh" / "id_ed25519.pub"
GPU_TYPE_ID = "NVIDIA A40"
DC_IDS = ["CA-MTL-1", "EU-SE-1"]  # HIGH-stock snapshot 2026-08-15 (EV-0025)

MUTATION = """
mutation createPod($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) {
    id
    costPerHr
    desiredStatus
    lastStatusChange
    machineId
  }
}
"""


def build_input(data_center_id: str, public_key: str, pod_name: str) -> dict:
    return {
        "cloudType": "SECURE",
        "containerDiskInGb": 30,
        "dataCenterId": data_center_id,
        "env": [{"key": "PUBLIC_KEY", "value": public_key}],
        "gpuCount": 1,
        "gpuTypeId": GPU_TYPE_ID,
        "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "minMemoryInGb": 20,
        "minVcpuCount": 4,
        "name": pod_name,
        "ports": "22/tcp",
        "startSsh": True,
        "templateId": "runpod-torch-v240",
        "volumeInGb": 50,
        "volumeMountPath": "/workspace",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pod-name", default=DEFAULT_POD_NAME)
    args = parser.parse_args()
    pod_name = args.pod_name

    key = load_key()
    pods = rest(key, "/pods")
    existing = [p for p in pods if isinstance(p, dict) and p.get("name") == pod_name]
    if existing:
        print(f"refusing duplicate; existing: {json.dumps(existing[0], default=str)[:300]}")
        return 0

    public_key = PUBLIC_KEY_PATH.read_text(encoding="utf-8").strip()
    if args.dry_run:
        sample = build_input(DC_IDS[0], "<redacted local public key>", pod_name)
        print(json.dumps({"mutation": "podFindAndDeployOnDemand", "input": sample}, indent=2))
        return 0

    before = account_snapshot(key)
    last_error = "no data centers tried"
    for dc_id in DC_IDS:
        payload = {
            "query": MUTATION,
            "variables": {"input": build_input(dc_id, public_key, pod_name)},
        }
        try:
            result = _http(GQL_API, key, payload=payload)
        except RuntimeError as exc:
            last_error = str(exc)
            print(f"{dc_id}: {last_error[:200]}", file=sys.stderr)
            continue
        errors = result.get("errors")
        if errors:
            last_error = json.dumps(errors)[:300]
            print(f"{dc_id}: {last_error}", file=sys.stderr)
            continue
        pod = result.get("data", {}).get("podFindAndDeployOnDemand")
        if not pod:
            last_error = f"{dc_id}: nil pod in response"
            continue
        record_billing(
            "POD_CREATE",
            {"id": pod.get("id"), "name": pod_name, "desiredStatus": pod.get("desiredStatus"),
             "costPerHr": pod.get("costPerHr"), "machineId": pod.get("machineId")},
            {
                "balance_before": before["clientBalance"],
                "gpu_type": GPU_TYPE_ID,
                "cloud_type": "SECURE",
                "data_center_id": dc_id,
                "predeclaration": "EV-0025 A40 fallback",
            },
        )
        print(json.dumps({"created": pod, "data_center_id": dc_id}, indent=2, sort_keys=True))
        return 0
    print(f"creation failed in all data centers: {last_error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
