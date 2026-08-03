"""Tiny recurrent MLP control architecture for pipeline tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from generals_bot.models.observation_encoder import (
    MAX_HW,
    NUM_CELL_CHANNELS,
    encode_globals,
    encode_observation,
)
from generals_bot.observation import Observation

# Actions: pass + build*H*W + move*H*W*4*2  (padded to MAX_HW)
NUM_DIRECTIONS = 4
NUM_SPLITS = 2
ACTION_DIM = 1 + (MAX_HW * MAX_HW) + (MAX_HW * MAX_HW * NUM_DIRECTIONS * NUM_SPLITS)


@dataclass
class MLPConfig:
    hidden: int = 128
    recurrent: int = 64
    global_dim: int = 9


class RecurrentMLPPolicy(nn.Module):
    def __init__(self, config: MLPConfig | None = None) -> None:
        super().__init__()
        self.config = config or MLPConfig()
        flat = NUM_CELL_CHANNELS * MAX_HW * MAX_HW
        self.encoder = nn.Sequential(
            nn.Linear(flat + self.config.global_dim, self.config.hidden),
            nn.ReLU(),
            nn.Linear(self.config.hidden, self.config.hidden),
            nn.ReLU(),
        )
        self.rnn = nn.GRUCell(self.config.hidden, self.config.recurrent)
        self.actor = nn.Linear(self.config.recurrent, ACTION_DIM)
        self.critic = nn.Linear(self.config.recurrent, 1)

    def initial_hidden(self, batch: int = 1, device: torch.device | None = None) -> Tensor:
        device = device or next(self.parameters()).device
        return torch.zeros(batch, self.config.recurrent, device=device)

    def forward_obs(
        self,
        observation: Observation,
        hidden: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        device = hidden.device
        cells = encode_observation(observation, device=device).reshape(1, -1)
        glob = encode_globals(observation, device=device).unsqueeze(0)
        x = self.encoder(torch.cat([cells, glob], dim=-1))
        h = self.rnn(x, hidden)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value, h

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
