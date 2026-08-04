"""Phase 9E GPU environment probe — nvidia-smi + torch CUDA snapshot."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from generals_bot.training.device_policy import cuda_runtime_snapshot, load_device_policy, resolve_training_device

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments/manifests/gpu_environment_probe.json"


def _nvidia_smi() -> dict:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,power.draw,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=30,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    gpus = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 8:
            continue
        gpus.append(
            {
                "name": parts[0],
                "driver_version": parts[1],
                "memory_total_mib": float(parts[2]),
                "memory_used_mib": float(parts[3]),
                "memory_free_mib": float(parts[4]),
                "temperature_c": float(parts[5]),
                "power_draw_w": float(parts[6]),
                "utilization_gpu_pct": float(parts[7]),
            }
        )
    return {"ok": True, "gpus": gpus, "raw": raw}


def main() -> int:
    policy = load_device_policy()
    snap = cuda_runtime_snapshot()
    smi = _nvidia_smi()
    device = None
    resolve_error = None
    try:
        device = resolve_training_device("auto", policy=policy, context="gpu_environment_probe")
        # Prove allocation + matmul on CUDA
        a = torch.randn(256, 256, device=device)
        b = torch.randn(256, 256, device=device)
        c = a @ b
        torch.cuda.synchronize()
        allocated = int(torch.cuda.memory_allocated())
        del a, b, c
        torch.cuda.empty_cache()
        matmul_ok = allocated > 0
    except Exception as exc:  # noqa: BLE001
        resolve_error = str(exc)
        matmul_ok = False
        allocated = 0

    gate = "PASS" if device and matmul_ok and snap["cuda_available"] else "FAIL"
    report = {
        "schema_version": 1,
        "kind": "GPU_ENVIRONMENT_PROBE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "device_policy": policy,
        "torch": snap,
        "nvidia_smi": smi,
        "resolved_device": device,
        "resolve_error": resolve_error,
        "cuda_matmul_probe_ok": matmul_ok,
        "cuda_memory_allocated_bytes_during_probe": allocated,
        "GPU_ENVIRONMENT_PROBE_GATE": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gate": gate, "device": device, "out": str(OUT)}, indent=2))
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
