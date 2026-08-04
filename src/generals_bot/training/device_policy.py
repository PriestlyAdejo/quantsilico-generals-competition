"""Fail-fast CUDA device resolution for neural training (Phase 9E)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY_PATH = REPO_ROOT / "configs" / "training" / "device_policy.yaml"


class DevicePolicyError(RuntimeError):
    """Raised when neural training cannot bind to the required CUDA device."""


def load_device_policy(path: Path | None = None) -> dict[str, Any]:
    policy_path = path or DEFAULT_POLICY_PATH
    if not policy_path.is_file():
        raise DevicePolicyError(f"device policy missing: {policy_path}")
    data = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise DevicePolicyError(f"device policy must be a mapping: {policy_path}")
    return data


def cuda_runtime_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_built": bool(torch.version.cuda),
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        idx = 0
        props = torch.cuda.get_device_properties(idx)
        snap.update(
            {
                "device_index": idx,
                "device_name": torch.cuda.get_device_name(idx),
                "total_memory_bytes": int(props.total_memory),
                "capability": f"{props.major}.{props.minor}",
            }
        )
    return snap


def resolve_training_device(
    requested: str | None = None,
    *,
    policy: dict[str, Any] | None = None,
    policy_path: Path | None = None,
    context: str = "neural_training",
) -> str:
    """Resolve neural-training device. Never silently falls back to CPU."""
    pol = policy if policy is not None else load_device_policy(policy_path)
    require_cuda = bool(pol.get("require_cuda_for_neural_training", True))
    allow_cpu = bool(pol.get("allow_cpu_training", False))
    fail_silent = bool(pol.get("fail_on_silent_cpu_fallback", True))
    default_cuda = str(pol.get("default_cuda_device", "cuda:0"))

    choice = (requested or "auto").strip().lower()
    if choice in {"auto", "cuda", "gpu"}:
        choice = default_cuda

    if choice.startswith("cuda"):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise DevicePolicyError(
                "CUDA required for neural training but torch.cuda.is_available() is False. "
                f"context={context} snapshot={cuda_runtime_snapshot()}"
            )
        # Validate index if present
        if ":" in choice:
            try:
                idx = int(choice.split(":", 1)[1])
            except ValueError as exc:
                raise DevicePolicyError(f"invalid CUDA device string: {choice}") from exc
            if idx < 0 or idx >= torch.cuda.device_count():
                raise DevicePolicyError(
                    f"CUDA device index out of range: {choice} "
                    f"(count={torch.cuda.device_count()})"
                )
        # Force a tiny allocation to catch broken runtime early
        try:
            probe = torch.zeros(1, device=choice)
            del probe
            torch.cuda.synchronize(device=choice)
        except Exception as exc:  # noqa: BLE001 — surface any CUDA init failure
            raise DevicePolicyError(
                f"CUDA device {choice} failed probe allocation: {exc}"
            ) from exc
        return choice

    if choice == "cpu":
        if require_cuda and not allow_cpu:
            if fail_silent:
                raise DevicePolicyError(
                    "CPU neural training forbidden by device_policy.yaml "
                    f"(require_cuda_for_neural_training=true). context={context}"
                )
        return "cpu"

    raise DevicePolicyError(f"unsupported training device request: {requested!r}")


def assert_module_on_cuda(module: torch.nn.Module, expected: str = "cuda:0") -> None:
    if not str(expected).startswith("cuda"):
        return
    try:
        param = next(module.parameters())
    except StopIteration as exc:
        raise DevicePolicyError("model has no parameters to verify device") from exc
    if param.device.type != "cuda":
        raise DevicePolicyError(
            f"model parameters on {param.device}, expected CUDA ({expected})"
        )
