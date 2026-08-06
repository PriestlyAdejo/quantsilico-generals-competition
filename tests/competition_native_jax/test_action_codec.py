"""Action codec roundtrip tests."""

from __future__ import annotations

from generals_bot.action import Action
from generals_bot.competition_native_jax.action_codec import action_to_index, index_to_action
from generals_bot.competition_native_jax.constants import ACTION_DIM, MAX_HW, PASS_INDEX


def test_action_dim() -> None:
    assert ACTION_DIM == 3970
    assert PASS_INDEX == 0


def test_pass_roundtrip() -> None:
    a = Action(kind=1)
    assert action_to_index(a) == 0
    assert index_to_action(0).kind == 1


def test_move_build_roundtrip_sample() -> None:
    for r, c in [(0, 0), (10, 10), (20, 20)]:
        for d in range(4):
            for s in range(2):
                a = Action(kind=0, row=r, col=c, direction=d, split=s)
                idx = action_to_index(a)
                b = index_to_action(idx)
                assert b == a
        a = Action(kind=2, row=r, col=c)
        assert index_to_action(action_to_index(a)) == a


def test_indices_cover_full_space() -> None:
    seen = set()
    seen.add(action_to_index(Action(kind=1)))
    for r in range(MAX_HW):
        for c in range(MAX_HW):
            for d in range(4):
                for s in range(2):
                    seen.add(action_to_index(Action(kind=0, row=r, col=c, direction=d, split=s)))
            seen.add(action_to_index(Action(kind=2, row=r, col=c)))
    assert len(seen) == ACTION_DIM
    assert min(seen) == 0
    assert max(seen) == ACTION_DIM - 1
