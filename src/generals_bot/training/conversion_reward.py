"""Minimal Phase 9D conversion reward — contact shaping + terminal integrity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from generals_bot.protocol import OWNER_OPP


@dataclass(frozen=True)
class TerminalRewardConfig:
    win: float = 1.0
    draw: float = 0.0
    loss: float = -1.0
    name: str = "win1_draw0_loss-1"

    def terminal_reward(self, *, winner: int | None, perspective: int = 0) -> float:
        if winner is None or winner < 0:
            return self.draw
        if winner == perspective:
            return self.win
        return self.loss

    def validate_ordering(self) -> None:
        if not (self.win > self.draw > self.loss):
            raise ValueError(
                f"Terminal ordering violated: win={self.win} draw={self.draw} loss={self.loss}"
            )


@dataclass(frozen=True)
class ContactShapingConfig:
    """Policy-visible contact incentive: newly observed enemy-owned cells."""

    enabled: bool = False
    bonus_per_new_enemy_cell: float = 0.05
    max_per_step: float = 0.15
    max_episode_cumulative: float = 0.6
    name: str = "contact_visible_enemy_delta"

    def step_bonus(self, *, prev_enemy_cells: int, curr_enemy_cells: int, episode_cum: float) -> float:
        if not self.enabled:
            return 0.0
        delta = max(0, curr_enemy_cells - prev_enemy_cells)
        raw = delta * self.bonus_per_new_enemy_cell
        capped = min(raw, self.max_per_step)
        room = max(0.0, self.max_episode_cumulative - episode_cum)
        return float(min(capped, room))


@dataclass(frozen=True)
class RewardConfig:
    """Versioned reward + curriculum for Phase 9D matched pilots."""

    version: str
    terminal: TerminalRewardConfig
    contact_shaping: ContactShapingConfig
    training_opponent: str  # "pass" | "official_expander"
    diagnosis: str
    primary_reward_family: str
    curriculum_mechanism: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "terminal": asdict(self.terminal),
            "contact_shaping": asdict(self.contact_shaping),
            "training_opponent": self.training_opponent,
            "diagnosis": self.diagnosis,
            "primary_reward_family": self.primary_reward_family,
            "curriculum_mechanism": self.curriculum_mechanism,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "RewardConfig":
        term = data.get("terminal") or {}
        contact = data.get("contact_shaping") or {}
        cfg = RewardConfig(
            version=str(data["version"]),
            terminal=TerminalRewardConfig(**{k: term[k] for k in ("win", "draw", "loss", "name") if k in term})
            if term
            else TerminalRewardConfig(),
            contact_shaping=ContactShapingConfig(
                **{
                    k: contact[k]
                    for k in (
                        "enabled",
                        "bonus_per_new_enemy_cell",
                        "max_per_step",
                        "max_episode_cumulative",
                        "name",
                    )
                    if k in contact
                }
            )
            if contact
            else ContactShapingConfig(),
            training_opponent=str(data.get("training_opponent", "pass")),
            diagnosis=str(data.get("diagnosis", "")),
            primary_reward_family=str(data.get("primary_reward_family", "none")),
            curriculum_mechanism=str(data.get("curriculum_mechanism", "none")),
        )
        cfg.terminal.validate_ordering()
        return cfg


CONTROL_V1 = RewardConfig(
    version="control_v1",
    terminal=TerminalRewardConfig(),
    contact_shaping=ContactShapingConfig(enabled=False),
    training_opponent="pass",
    diagnosis="N/A",
    primary_reward_family="sparse_terminal",
    curriculum_mechanism="pass_opponent",
)

# Shared: both CNN and graph diagnosed CONTACT_FAILURE (+ DISCOVERY_FAILURE secondary).
CONVERSION_V1 = RewardConfig(
    version="conversion_v1",
    terminal=TerminalRewardConfig(),  # unchanged terminal ordering
    contact_shaping=ContactShapingConfig(enabled=True),
    training_opponent="official_expander",
    diagnosis="CONTACT_FAILURE",
    primary_reward_family="contact_visible_enemy_delta",
    curriculum_mechanism="train_vs_official_expander",
)


def count_visible_enemy_cells(owner_grid: np.ndarray) -> int:
    return int(np.sum(np.asarray(owner_grid) == OWNER_OPP))


def assert_no_privileged_keys(reward_inputs: dict[str, Any]) -> None:
    banned = {
        "enemy_general_absolute",
        "hidden_opponent_army_total",
        "privileged_board",
        "ground_truth_general",
    }
    bad = banned.intersection(reward_inputs)
    if bad:
        raise ValueError(f"Privileged reward inputs forbidden: {sorted(bad)}")
