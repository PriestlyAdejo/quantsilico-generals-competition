"""Recurrent residual CNN — v1 flat actor + v2 factorised spatial actor."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from generals_bot.models.action_index import ACTION_DIM
from generals_bot.models.factorised_action import FactorisedActionHead
from generals_bot.models.heads import AuxiliaryHeads, HeadConfig, StrategicMixtureGate
from generals_bot.models.observation_encoder import (
    GLOBAL_DIM,
    NUM_CELL_CHANNELS,
    encode_globals,
    encode_observation,
)
from generals_bot.observation import Observation


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.net(x))


@dataclass
class CNNConfig:
    channels: int = 48
    blocks: int = 3
    recurrent: int = 96
    global_dim: int = GLOBAL_DIM
    architecture: str = "recurrent_cnn_v2"
    action_head: str = "factorised_spatial_v2"
    schema_version: int = 2


def _is_factorised(cfg: CNNConfig) -> bool:
    return cfg.architecture.endswith("_v2") or cfg.action_head.startswith("factorised")


class RecurrentCNNPolicy(nn.Module):
    def __init__(self, config: CNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or CNNConfig()
        if self.config.architecture == "recurrent_cnn_v1":
            self.config.action_head = "flat_linear_v1"
            self.config.schema_version = 1
        c = self.config.channels
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_CELL_CHANNELS, c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(c) for _ in range(self.config.blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fuse = nn.Sequential(
            nn.Linear(c + self.config.global_dim, self.config.recurrent),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.GRUCell(self.config.recurrent, self.config.recurrent)
        self.mixture = StrategicMixtureGate(self.config.recurrent)
        emb = self.mixture.option_embed.embedding_dim
        self.factorised = _is_factorised(self.config)
        if self.factorised:
            self.spatial_proj = nn.Conv2d(c, c, kernel_size=1)
            self.actor = FactorisedActionHead(c, context_dim=self.config.recurrent + emb)
        else:
            self.actor = nn.Linear(self.config.recurrent + emb, ACTION_DIM)
        self.aux = AuxiliaryHeads(HeadConfig(recurrent=self.config.recurrent))

    def initial_hidden(self, batch: int = 1, device: torch.device | None = None) -> Tensor:
        device = device or next(self.parameters()).device
        return torch.zeros(batch, self.config.recurrent, device=device)

    def forward_tensors(
        self,
        cells: Tensor,
        globals_: Tensor,
        hidden: Tensor,
        *,
        deterministic: bool = True,
        previous_option: Tensor | None = None,
        teacher_option: Tensor | None = None,
    ) -> dict[str, Tensor]:
        x = self.stem(cells)
        x = self.blocks(x)
        pooled = self.pool(x).flatten(1)
        fused = self.fuse(torch.cat([pooled, globals_], dim=-1))
        h = self.rnn(fused, hidden)
        mix_probs, opt_idx, opt_emb, mix_logits = self.mixture(
            h,
            previous_option=previous_option,
            deterministic=deterministic,
            teacher_option=teacher_option,
        )
        context = torch.cat([h, opt_emb], dim=-1)
        aux = self.aux(h)
        result: dict[str, Tensor] = {
            "value": aux["value"],
            "hidden": h,
            "mixture_probs": mix_probs,
            "mixture_logits": mix_logits,
            "option_index": opt_idx,
            "general_loss_risk": aux["general_loss_risk"],
            "belief": aux["belief"],
            "opponent_style": aux["opponent_style"],
            "concepts": aux["concepts"],
        }
        if self.factorised:
            spatial = self.spatial_proj(x)
            actor_out = self.actor(spatial, context)
            result.update(actor_out)
            result["spatial_features"] = spatial
        else:
            result["logits"] = self.actor(context)
        return result

    def forward_obs(
        self,
        observation: Observation,
        hidden: Tensor,
        *,
        deterministic: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        device = hidden.device
        cells = encode_observation(observation, device=device).unsqueeze(0)
        glob = encode_globals(observation, device=device).unsqueeze(0)
        out = self.forward_tensors(cells, glob, hidden, deterministic=deterministic)
        return out["logits"], out["value"], out["hidden"]

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config_dict(self) -> dict:
        return asdict(self.config)
