"""Safetensors checkpoint save/load with versioned architecture config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

try:
    from safetensors.torch import load_file, save_file
except ImportError:  # pragma: no cover - exercised in environments without safetensors
    load_file = None  # type: ignore[assignment]
    save_file = None  # type: ignore[assignment]


CHECKPOINT_SCHEMA_VERSION = 1


def save_checkpoint(
    model: nn.Module,
    path: Path | str,
    *,
    architecture: str,
    config: dict[str, Any],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights_path = path.with_suffix(".safetensors")
    config_path = path.with_suffix(".json")
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    if save_file is None:
        torch.save(state, path.with_suffix(".pt"))
        weights_path = path.with_suffix(".pt")
    else:
        save_file(state, str(weights_path))
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "architecture": architecture,
        "config": config,
        "weights": weights_path.name,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
    }
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return config_path


def load_checkpoint_payload(config_path: Path | str) -> dict[str, Any]:
    config_path = Path(config_path)
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_state_dict(config_path: Path | str, map_location: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    config_path = Path(config_path)
    payload = load_checkpoint_payload(config_path)
    weights = config_path.parent / payload["weights"]
    if weights.suffix == ".safetensors":
        if load_file is None:
            raise RuntimeError("safetensors is required to load this checkpoint")
        return load_file(str(weights), device=str(map_location))
    return torch.load(weights, map_location=map_location, weights_only=True)


def apply_state_dict(model: nn.Module, config_path: Path | str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = load_checkpoint_payload(config_path)
    state = load_state_dict(config_path, map_location=map_location)
    model.load_state_dict(state)
    return payload
