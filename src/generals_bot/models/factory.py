"""Model factory helpers for versioned architectures."""

from __future__ import annotations

from typing import Any

from torch import nn

from generals_bot.models.cnn import CNNConfig, RecurrentCNNPolicy
from generals_bot.models.graph import GraphConfig, RecurrentGraphBeliefPolicy
from generals_bot.models.mlp import MLPConfig, RecurrentMLPPolicy

_ARCH = {
    "recurrent_mlp_v1": (RecurrentMLPPolicy, MLPConfig),
    "recurrent_cnn_v1": (RecurrentCNNPolicy, CNNConfig),
    "recurrent_graph_belief_v1": (RecurrentGraphBeliefPolicy, GraphConfig),
}


def build_model(architecture: str, config: dict[str, Any] | None = None) -> nn.Module:
    if architecture not in _ARCH:
        raise KeyError(f"unknown architecture: {architecture}")
    cls, cfg_cls = _ARCH[architecture]
    cfg = cfg_cls(**{k: v for k, v in (config or {}).items() if k in cfg_cls.__dataclass_fields__})
    return cls(cfg)


def known_architectures() -> list[str]:
    return sorted(_ARCH)
