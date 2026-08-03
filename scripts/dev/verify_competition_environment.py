#!/usr/bin/env python3
"""Verify competition (.venv) environment parity signals."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

EXPECTED = {
    "numpy": "2.4.6",
    "torch": "2.13.0",
    "jax": "0.11.0",
    "safetensors": "0.8.0",
    "gymnasium": "1.3.0",
}


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"executable: {sys.executable}")
    print(f"platform: {platform.platform()}")
    if not sys.version.startswith("3.12.10"):
        print("FAIL: expected Python 3.12.10")
        return 1

    ok = True
    for mod, want in EXPECTED.items():
        try:
            m = importlib.import_module(mod)
            got = getattr(m, "__version__", "?")
            print(f"{mod}: {got}")
            if not str(got).startswith(want):
                print(f"FAIL: {mod} expected {want}*, got {got}")
                ok = False
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: {mod} import: {exc}")
            ok = False

    try:
        import torch

        if torch.cuda.is_available():
            print("WARN: competition venv has CUDA available; expected CPU-only torch")
        print(f"torch cuda available: {torch.cuda.is_available()}")
    except Exception:
        pass

    try:
        import generals  # noqa: F401

        import generals_bot

        print(f"generals_bot: {generals_bot.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: package import: {exc}")
        ok = False

    seeds = Path("experiments/seeds/MANIFEST.sha256")
    if not seeds.exists():
        print("FAIL: missing experiments/seeds/MANIFEST.sha256")
        ok = False
    else:
        print(f"seed manifest: {seeds}")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
