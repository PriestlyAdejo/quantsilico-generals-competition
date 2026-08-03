#!/usr/bin/env python3
"""Report the local Generals competition development environment."""

from __future__ import annotations

import importlib
import platform
import sys
from typing import Any


def _status(name: str, value: Any) -> None:
    print(f"{name}: {value}")


def _try_import(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - report any import failure
        return False, f"MISSING ({type(exc).__name__}: {exc})"
    version = getattr(module, "__version__", "unknown")
    return True, str(version)


def main() -> int:
    _status("Python", sys.version.replace("\n", " "))
    _status("executable", sys.executable)
    _status("platform", platform.platform())

    ok_numpy, numpy_v = _try_import("numpy")
    _status("NumPy", numpy_v if ok_numpy else numpy_v)

    ok_jax, jax_v = _try_import("jax")
    _status("JAX", jax_v if ok_jax else jax_v)
    if ok_jax:
        try:
            import jax

            backend = jax.default_backend()
            devices = jax.devices()
            _status("JAX backend", backend)
            _status("JAX devices", devices)
        except Exception as exc:  # noqa: BLE001
            _status("JAX backend", f"ERROR ({exc})")
    else:
        _status("JAX backend", "n/a")

    ok_torch, torch_v = _try_import("torch")
    _status("Torch", torch_v if ok_torch else torch_v)
    if ok_torch:
        try:
            import torch

            _status("Torch CUDA available", torch.cuda.is_available())
            if torch.cuda.is_available():
                _status("Torch CUDA device", torch.cuda.get_device_name(0))
            else:
                _status("Torch CUDA status", "CPU-only / unavailable")
        except Exception as exc:  # noqa: BLE001
            _status("Torch CUDA status", f"ERROR ({exc})")
    else:
        _status("Torch CUDA status", "n/a")

    ok_gym, gym_v = _try_import("gymnasium")
    _status("Gymnasium", gym_v if ok_gym else gym_v)

    ok_nx, nx_v = _try_import("networkx")
    _status("NetworkX", nx_v if ok_nx else nx_v)

    ok_engine, engine_msg = _try_import("generals")
    _status("official engine import", "ok" if ok_engine else engine_msg)

    ok_private, private_msg = _try_import("generals_bot")
    if ok_private:
        import generals_bot

        _status("private package import", f"ok ({generals_bot.__version__})")
    else:
        _status("private package import", private_msg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
