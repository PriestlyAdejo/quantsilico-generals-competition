#!/usr/bin/env python3
"""Detect training hardware and report CUDA readiness."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _nvidia_smi() -> dict:
    if not shutil.which("nvidia-smi"):
        return {"available": False, "error": "nvidia-smi not found"}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "available": True,
            "name": parts[0] if parts else None,
            "memory_total_mib": parts[1] if len(parts) > 1 else None,
            "driver_version": parts[2] if len(parts) > 2 else None,
            "temperature_c": parts[3] if len(parts) > 3 else None,
            "power_draw_w": parts[4] if len(parts) > 4 else None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}


def main() -> int:
    report: dict = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "nvidia_smi": _nvidia_smi(),
        "torch": None,
        "cuda_ok": False,
    }
    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        if torch.cuda.is_available():
            x = torch.zeros(1, device="cuda")
            y = x + 1
            report["cuda_ok"] = bool(y.item() == 1.0)
            report["torch"]["smoke"] = "ok" if report["cuda_ok"] else "fail"
    except Exception as exc:  # noqa: BLE001
        report["torch"] = {"error": str(exc)}

    out_dir = Path("var/training")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "hardware_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {path}")
    # CUDA failure is recorded, not a process failure for Phase 0.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
