"""Tiny recurrent MLP control architecture for pipeline tests."""

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
)
from generals_bot.observation import Observation


@dataclass
class MLPConfig:
    hidden: int = 128
    recurrent: int = 64
    global_dim: int = GLOBAL_DIM
    architecture: str = "recurrent_mlp_v1"
    action_head: str = "flat_linear_v1"
    schema_version: int = 1
    limitation: str = (
        "MLP uses a flat absolute ACTION_DIM linear actor; "
        "it is a weak control and does not share spatial parameters."
    )


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
        self.mixture = StrategicMixtureGate(self.config.recurrent, extra_dim=0)
        self.actor = nn.Linear(self.config.recurrent + self.mixture.option_embed.embedding_dim, ACTION_DIM)
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
        """Batched forward from pre-encoded tensors.

        cells: (B, C, H, W) or (B, C*H*W)
        globals_: (B, G)
        """
        if cells.ndim == 4:
            cells = cells.reshape(cells.shape[0], -1)
        x = self.encoder(torch.cat([cells, globals_], dim=-1))
        h = self.rnn(x, hidden)
        mix_probs, opt_idx, opt_emb, mix_logits = self.mixture(
            h,
            previous_option=previous_option,
            deterministic=deterministic,
            teacher_option=teacher_option,
        )
        logits = self.actor(torch.cat([h, opt_emb], dim=-1))
        aux = self.aux(h)
        return {
            "logits": logits,
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

    def forward_obs(
        self,
        observation: Observation,
        hidden: Tensor,
        *,
        deterministic: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        device = hidden.device
        cells = encode_observation(observation, device=device).reshape(1, -1)
        glob = encode_globals(observation, device=device).unsqueeze(0)
        out = self.forward_tensors(cells, glob, hidden, deterministic=deterministic)
        return out["logits"], out["value"], out["hidden"]

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config_dict(self) -> dict:
        return asdict(self.config)
