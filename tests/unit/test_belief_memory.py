"""Belief memory lifecycle tests."""

from __future__ import annotations

from generals_bot.core.belief import BELIEF_SCHEMA_VERSION, BeliefMemory, belief_channels_summary
from generals_bot.observation import Observation
from generals_bot.protocol import OWNER_ME, TYPE_PLAIN


def _obs(h: int = 4, w: int = 4) -> Observation:
    tg = tuple(tuple(TYPE_PLAIN for _ in range(w)) for _ in range(h))
    og = tuple(tuple(OWNER_ME if (r + c) % 2 == 0 else 0 for c in range(w)) for r in range(h))
    ag = tuple(tuple(1 for _ in range(w)) for _ in range(h))
    return Observation(
        height=h,
        width=w,
        turn=1,
        my_land=8,
        my_army=8,
        opp_land=0,
        opp_army=0,
        type_grid=tg,
        owner_grid=og,
        army_grid=ag,
    )


def test_belief_create_and_update():
    b = BeliefMemory.create(4, 4)
    assert b.schema_version == BELIEF_SCHEMA_VERSION
    b.update_visible(_obs())
    summary = belief_channels_summary(b)
    assert summary["coverage_frac"] > 0.0
