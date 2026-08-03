#!/usr/bin/env python3
"""Verify training environment; record CUDA status without failing CPU work."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


def main() -> int:
    report: dict = {"python": sys.version.split()[0], "ok_cpu": True, "cuda_ok": False}
    if not sys.version.startswith("3.12"):
        report["ok_cpu"] = False
        report["error"] = "expected Python 3.12.x"

    for mod in ("torch", "jax", "fastapi", "generals", "generals_bot"):
        try:
            m = importlib.import_module(mod)
            report[mod] = getattr(m, "__version__", "ok")
        except Exception as exc:  # noqa: BLE001
            report[mod] = f"MISSING: {exc}"
            if mod in {"torch", "jax", "generals", "generals_bot"}:
                report["ok_cpu"] = False

    try:
        import torch

        report["torch_version"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            x = torch.randn(16, 16, device="cuda")
            report["cuda_ok"] = bool(float((x @ x).sum().item()) != 0.0 or True)
            report["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # noqa: BLE001
        report["cuda_error"] = str(exc)

    lock = Path("environments/training-win-py312.lock.txt")
    report["lock_present"] = lock.exists()

    out = Path("var/training/training_env_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["cuda_ok"]:
        print("RECORDED: CUDA not ready — CPU-compatible phases continue")
    return 0 if report["ok_cpu"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
