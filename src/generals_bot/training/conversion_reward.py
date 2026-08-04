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
    # Phase 9E curriculum: one-shot discovery bonus + gate contact until discovery.
    discovery_bonus_once: float = 0.0
    gate_contact_until_discovery: bool = False

    def step_bonus(
        self,
        *,
        prev_enemy_cells: int,
        curr_enemy_cells: int,
        episode_cum: float,
        discovered: bool,
    ) -> tuple[float, bool]:
        """Return (bonus, discovered_after)."""
        if not self.enabled:
            return 0.0, discovered or curr_enemy_cells > 0
        newly_discovered = (not discovered) and curr_enemy_cells > 0
        discovered_after = discovered or curr_enemy_cells > 0
        bonus = 0.0
        if newly_discovered and self.discovery_bonus_once > 0:
            room = max(0.0, self.max_episode_cumulative - episode_cum)
            bonus += float(min(self.discovery_bonus_once, room))
        if self.gate_contact_until_discovery and not discovered_after:
            return bonus, discovered_after
        # After discovery (or if ungated), apply visible-enemy delta shaping.
        if self.gate_contact_until_discovery and newly_discovered:
            # Contact delta on the discovery step uses prev=0 → curr; allow it.
            pass
        delta = max(0, curr_enemy_cells - prev_enemy_cells)
        raw = delta * self.bonus_per_new_enemy_cell
        capped = min(raw, self.max_per_step)
        room = max(0.0, self.max_episode_cumulative - episode_cum - bonus)
        bonus += float(min(capped, room))
        return bonus, discovered_after


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
                        "discovery_bonus_once",
                        "gate_contact_until_discovery",
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

# Phase 9E curriculum-only: discovery-gated contact (one reward family), vs Expander.
CURRICULUM_DISCOVERY_V1 = RewardConfig(
    version="curriculum_discovery_v1",
    terminal=TerminalRewardConfig(),
    contact_shaping=ContactShapingConfig(
        enabled=True,
        bonus_per_new_enemy_cell=0.05,
        max_per_step=0.15,
        max_episode_cumulative=0.6,
        name="discovery_gated_visible_enemy",
        discovery_bonus_once=0.2,
        gate_contact_until_discovery=True,
    ),
    training_opponent="official_expander",
    diagnosis="CONTACT_FAILURE+DISCOVERY_FAILURE",
    primary_reward_family="discovery_gated_visible_enemy",
    curriculum_mechanism="discovery_gate_then_contact_vs_expander",
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
