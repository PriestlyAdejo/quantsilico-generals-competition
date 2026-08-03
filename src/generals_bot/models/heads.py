"""Shared actor–critic auxiliary heads and strategic mixture gate."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

STRATEGIC_OPTIONS: tuple[str, ...] = (
    "WAIT",
    "SCOUT",
    "EXPAND",
    "COLLECT",
    "DEFEND",
    "BUILD",
    "PRESSURE",
    "ATTACK",
    "GENERAL_HUNT",
    "DEATHTOUCH",
    "ENDGAME",
)
NUM_OPTIONS = len(STRATEGIC_OPTIONS)

OPPONENT_ARCHETYPES: tuple[str, ...] = (
    "UNKNOWN",
    "RAPID_EXPANDER",
    "EARLY_AGGRESSOR",
    "COLLECTOR",
    "CASTLE_INVESTOR",
    "DEFENSIVE_TURTLE",
    "GENERAL_HUNTER",
    "DEATHTOUCH_SPECIALIST",
    "MIXED",
)
NUM_ARCHETYPES = len(OPPONENT_ARCHETYPES)
UNKNOWN_FLOOR = 0.05


@dataclass
class HeadConfig:
    recurrent: int = 64
    option_embed: int = 16
    belief_dim: int = 8
    opponent_dim: int = NUM_ARCHETYPES


class StrategicMixtureGate(nn.Module):
    """Soft mixture over strategic options; option-conditions the action head."""

    def __init__(self, recurrent: int, extra_dim: int = 0, option_embed: int = 16) -> None:
        super().__init__()
        self.option_embed = nn.Embedding(NUM_OPTIONS, option_embed)
        in_dim = recurrent + extra_dim + option_embed
        self.gate = nn.Sequential(
            nn.Linear(in_dim, recurrent),
            nn.ReLU(),
            nn.Linear(recurrent, NUM_OPTIONS),
        )

    def forward(
        self,
        recurrent_state: Tensor,
        extras: Tensor | None = None,
        previous_option: Tensor | None = None,
        *,
        deterministic: bool = False,
        teacher_option: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (mixture_probs, option_indices, option_embeddings, gate_logits)."""
        batch = recurrent_state.shape[0]
        device = recurrent_state.device
        if previous_option is None:
            previous_option = torch.zeros(batch, dtype=torch.long, device=device)
        prev_emb = self.option_embed(previous_option)
        parts = [recurrent_state, prev_emb]
        if extras is not None:
            parts.insert(1, extras)
        logits = self.gate(torch.cat(parts, dim=-1))
        probs = F.softmax(logits, dim=-1)
        if teacher_option is not None:
            indices = teacher_option.long()
        elif deterministic:
            indices = torch.argmax(probs, dim=-1)
        else:
            indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
        emb = self.option_embed(indices)
        return probs, indices, emb, logits


class OpponentStyleHead(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, NUM_ARCHETYPES))

    def forward(self, x: Tensor) -> Tensor:
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        # Enforce UNKNOWN probability floor then renormalise.
        floor = UNKNOWN_FLOOR
        probs = probs.clone()
        probs[..., 0] = torch.clamp(probs[..., 0], min=floor)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs


class AuxiliaryHeads(nn.Module):
    def __init__(self, config: HeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or HeadConfig()
        r = self.config.recurrent
        self.value = nn.Linear(r, 1)
        self.general_loss_risk = nn.Linear(r, 1)
        self.belief = nn.Linear(r, self.config.belief_dim)
        self.opponent = OpponentStyleHead(r)
        self.concepts = nn.Linear(r, 4)

    def forward(self, h: Tensor) -> dict[str, Tensor]:
        return {
            "value": self.value(h).squeeze(-1),
            "general_loss_risk": torch.sigmoid(self.general_loss_risk(h).squeeze(-1)),
            "belief": self.belief(h),
            "opponent_style": self.opponent(h),
            "concepts": self.concepts(h),
        }
