"""Reward-integrity tests for Phase 9D conversion configs."""

from __future__ import annotations

import pytest
import yaml
from pathlib import Path

from generals_bot.training.conversion_reward import (
    CONVERSION_V1,
    CONTROL_V1,
    RewardConfig,
    TerminalRewardConfig,
    assert_no_privileged_keys,
)

REPO = Path(__file__).resolve().parents[1]


def test_terminal_ordering_win_gt_draw_gt_loss() -> None:
    CONTROL_V1.terminal.validate_ordering()
    CONVERSION_V1.terminal.validate_ordering()
    with pytest.raises(ValueError):
        TerminalRewardConfig(win=1.0, draw=-1.0, loss=-0.2).validate_ordering()


def test_shaping_upper_bounds() -> None:
    c = CONVERSION_V1.contact_shaping
    assert c.enabled
    ep = 0.0
    discovered = False
    for _ in range(100):
        b, discovered = c.step_bonus(
            prev_enemy_cells=0, curr_enemy_cells=10, episode_cum=ep, discovered=discovered
        )
        assert b <= c.max_per_step + 1e-9
        ep += b
    assert ep <= c.max_episode_cumulative + 1e-9
    assert ep < CONTROL_V1.terminal.win


def test_discovery_gated_curriculum_fires_once() -> None:
    from generals_bot.training.conversion_reward import CURRICULUM_DISCOVERY_V1

    c = CURRICULUM_DISCOVERY_V1.contact_shaping
    b0, d0 = c.step_bonus(prev_enemy_cells=0, curr_enemy_cells=0, episode_cum=0.0, discovered=False)
    assert b0 == 0.0 and d0 is False
    b1, d1 = c.step_bonus(prev_enemy_cells=0, curr_enemy_cells=3, episode_cum=0.0, discovered=False)
    assert d1 is True
    assert b1 >= c.discovery_bonus_once
    b2, d2 = c.step_bonus(prev_enemy_cells=3, curr_enemy_cells=3, episode_cum=b1, discovered=True)
    assert d2 is True
    assert b2 == 0.0


def test_no_privileged_state_access() -> None:
    assert_no_privileged_keys({"owner_grid_visible": True})
    with pytest.raises(ValueError):
        assert_no_privileged_keys({"enemy_general_absolute": (1, 2)})


def test_config_serialize_roundtrip() -> None:
    d = CONVERSION_V1.to_dict()
    again = RewardConfig.from_dict(d)
    assert again.to_dict() == d


def test_yaml_configs_load() -> None:
    for name in ("control_v1.yaml", "conversion_v1.yaml"):
        path = REPO / "configs/training/draw_conversion" / name
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg = RewardConfig.from_dict(data)
        cfg.terminal.validate_ordering()
