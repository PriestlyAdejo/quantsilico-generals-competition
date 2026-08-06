"""Legal mask includes PASS."""

from __future__ import annotations

from generals_bot.competition_native_jax.legal_mask import legal_mask_from_observation
from generals_bot.observation import Observation


def _empty_obs(h: int = 18, w: int = 18) -> Observation:
    z = tuple(tuple(0 for _ in range(w)) for _ in range(h))
    return Observation(
        height=h,
        width=w,
        turn=0,
        my_land=0,
        my_army=0,
        opp_land=0,
        opp_army=0,
        type_grid=z,
        owner_grid=z,
        army_grid=z,
    )


def test_pass_always_legal() -> None:
    mask = legal_mask_from_observation(_empty_obs())
    assert mask[0]
