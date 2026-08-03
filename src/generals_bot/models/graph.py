"""Recurrent graph-belief actor–critic using pure PyTorch tensor shifts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from generals_bot.models.action_index import ACTION_DIM
from generals_bot.models.heads import AuxiliaryHeads, HeadConfig, StrategicMixtureGate
from generals_bot.models.observation_encoder import (
    GLOBAL_DIM,
    MAX_HW,
    NUM_CELL_CHANNELS,
    encode_globals,
    encode_observation,
    neighbour_index_tables,
)
from generals_bot.observation import Observation


def _shift(x: Tensor, direction: str) -> Tensor:
    """Shift spatial tensor (B, C, H, W) so each cell receives its neighbour.

    Pad spec for 4D tensors is (left, right, top, bottom).
    """
    if direction == "self":
        return x
    if direction == "north":
        return nn.functional.pad(x[:, :, :-1, :], (0, 0, 1, 0))
    if direction == "south":
        return nn.functional.pad(x[:, :, 1:, :], (0, 0, 0, 1))
    if direction == "west":
        return nn.functional.pad(x[:, :, :, :-1], (1, 0, 0, 0))
    if direction == "east":
        return nn.functional.pad(x[:, :, :, 1:], (0, 1, 0, 0))
    raise ValueError(direction)


class DirectionalMessageBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dirs = ("north", "south", "east", "west", "self")
        self.proj = nn.ModuleDict({d: nn.Conv2d(channels, channels, kernel_size=1) for d in self.dirs})
        self.combine = nn.Sequential(
            nn.Conv2d(channels * len(self.dirs), channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        msgs = [self.proj[d](_shift(x, d)) for d in self.dirs]
        out = self.combine(torch.cat(msgs, dim=1))
        return self.act(x + out)


@dataclass
class GraphConfig:
    channels: int = 48
    recurrent_channels: int = 24
    global_dim: int = GLOBAL_DIM
    recurrent: int = 96
    architecture: str = "recurrent_graph_belief_v1"


class RecurrentGraphBeliefPolicy(nn.Module):
    def __init__(self, config: GraphConfig | None = None) -> None:
        super().__init__()
        self.config = config or GraphConfig()
        c = self.config.channels
        rc = self.config.recurrent_channels
        self.input_proj = nn.Sequential(
            nn.Conv2d(NUM_CELL_CHANNELS, c, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.pre = nn.Sequential(DirectionalMessageBlock(c), DirectionalMessageBlock(c))
        self.cell_gate = nn.Conv2d(c + rc, rc, kernel_size=1)
        self.cell_cand = nn.Conv2d(c + rc, rc, kernel_size=1)
        self.post = nn.Sequential(DirectionalMessageBlock(rc), DirectionalMessageBlock(rc))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fuse = nn.Sequential(
            nn.Linear(rc + self.config.global_dim, self.config.recurrent),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.GRUCell(self.config.recurrent, self.config.recurrent)
        self.mixture = StrategicMixtureGate(self.config.recurrent)
        emb = self.mixture.option_embed.embedding_dim
        self.actor = nn.Linear(self.config.recurrent + emb, ACTION_DIM)
        self.aux = AuxiliaryHeads(HeadConfig(recurrent=self.config.recurrent))
        # Warm static topology cache (deterministic).
        neighbour_index_tables()

    def initial_hidden(self, batch: int = 1, device: torch.device | None = None) -> Tensor:
        device = device or next(self.parameters()).device
        return torch.zeros(batch, self.config.recurrent, device=device)

    def initial_cell_memory(self, batch: int = 1, device: torch.device | None = None) -> Tensor:
        device = device or next(self.parameters()).device
        return torch.zeros(
            batch,
            self.config.recurrent_channels,
            MAX_HW,
            MAX_HW,
            device=device,
        )

    def forward_tensors(
        self,
        cells: Tensor,
        globals_: Tensor,
        hidden: Tensor,
        cell_memory: Tensor | None = None,
        *,
        deterministic: bool = True,
        previous_option: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch = cells.shape[0]
        device = cells.device
        if cell_memory is None:
            cell_memory = self.initial_cell_memory(batch, device=device)
        x = self.input_proj(cells)
        x = self.pre(x)
        gate_in = torch.cat([x, cell_memory], dim=1)
        gate = torch.sigmoid(self.cell_gate(gate_in))
        cand = torch.tanh(self.cell_cand(gate_in))
        cell_memory = gate * cell_memory + (1.0 - gate) * cand
        x = self.post(cell_memory)
        pooled = self.pool(x).flatten(1)
        fused = self.fuse(torch.cat([pooled, globals_], dim=-1))
        h = self.rnn(fused, hidden)
        mix_probs, opt_idx, opt_emb = self.mixture(
            h, previous_option=previous_option, deterministic=deterministic
        )
        logits = self.actor(torch.cat([h, opt_emb], dim=-1))
        aux = self.aux(h)
        return {
            "logits": logits,
            "value": aux["value"],
            "hidden": h,
            "cell_memory": cell_memory,
            "mixture_probs": mix_probs,
            "option_index": opt_idx,
            "general_loss_risk": aux["general_loss_risk"],
            "belief": aux["belief"],
            "opponent_style": aux["opponent_style"],
            "concepts": aux["concepts"],
        }

    def forward_obs(
        self,
        observation: Observation,
        hidden: Tensor,
        cell_memory: Tensor | None = None,
        *,
        deterministic: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        device = hidden.device
        cells = encode_observation(observation, device=device).unsqueeze(0)
        glob = encode_globals(observation, device=device).unsqueeze(0)
        out = self.forward_tensors(
            cells, glob, hidden, cell_memory, deterministic=deterministic
        )
        return out["logits"], out["value"], out["hidden"], out["cell_memory"]

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config_dict(self) -> dict:
        return asdict(self.config)
