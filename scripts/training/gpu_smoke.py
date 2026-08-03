#!/usr/bin/env python3
"""Minimal CUDA smoke for training venv. Exit 0 even on CUDA fail (recorded)."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    result = {"cuda_ok": False}
    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            a = torch.randn(1024, 1024, device="cuda")
            b = a @ a.T
            result["checksum"] = float(b[0, 0].item())
            result["cuda_ok"] = True
            result["device"] = torch.cuda.get_device_name(0)
        else:
            result["error"] = "torch.cuda.is_available() is False"
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    path = Path("var/training/gpu_smoke.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("PASS" if result["cuda_ok"] else "RECORDED_FAIL_CUDA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
