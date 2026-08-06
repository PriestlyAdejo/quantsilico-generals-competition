"""PPO zero-update ratio gate."""

from __future__ import annotations

import numpy as np

from generals_bot.competition_native_jax.constants import ACTION_DIM
from train.competition_native_jax.ppo import assert_zero_update_ratio


def test_zero_update_rho_is_one() -> None:
    logits = np.linspace(-1, 1, ACTION_DIM)
    mask = np.zeros(ACTION_DIM, dtype=bool)
    mask[0] = True
    mask[10:20] = True
    rho = assert_zero_update_ratio(logits, mask, 0)
    assert abs(rho - 1.0) < 1e-9
