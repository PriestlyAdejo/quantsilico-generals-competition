"""Property tests for legal-action generation."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from generals_bot.action import PASS_ACTION
from generals_bot.legal import enumerate_legal_actions, is_legal_action
from generals_bot.observation import Observation
from generals_bot.protocol import TYPE_MOUNTAIN, TYPE_PLAIN, TYPE_STRUCTURE_IN_FOG


@st.composite
def small_observations(draw: st.DrawFn) -> Observation:
    h = draw(st.integers(2, 5))
    w = draw(st.integers(2, 5))
    types = []
    owners = []
    armies = []
    for _r in range(h):
        trow = []
        orow = []
        arow = []
        for _c in range(w):
            cell_type = draw(
                st.sampled_from([TYPE_PLAIN, TYPE_MOUNTAIN, TYPE_STRUCTURE_IN_FOG, 0, 1, 3, 4])
            )
            owner = draw(st.sampled_from([0, 1, 2]))
            # Own cells should not be impassable types for ownership consistency
            if owner == 1 and cell_type in (TYPE_MOUNTAIN, TYPE_STRUCTURE_IN_FOG, 0):
                cell_type = TYPE_PLAIN if draw(st.booleans()) else 4
            army = draw(st.integers(0, 60)) if owner == 1 else draw(st.integers(0, 20))
            trow.append(cell_type)
            orow.append(owner)
            arow.append(army)
        types.append(trow)
        owners.append(orow)
        armies.append(arow)
    # Ensure at least one owned cell
    types[0][0], owners[0][0], armies[0][0] = 4, 1, draw(st.integers(1, 40))
    return Observation(
        height=h,
        width=w,
        turn=draw(st.integers(0, 100)),
        my_land=sum(1 for row in owners for v in row if v == 1),
        my_army=sum(
            armies[r][c] for r in range(h) for c in range(w) if owners[r][c] == 1
        ),
        opp_land=1,
        opp_army=1,
        type_grid=tuple(tuple(r) for r in types),
        owner_grid=tuple(tuple(r) for r in owners),
        army_grid=tuple(tuple(r) for r in armies),
    )


@given(small_observations())
@settings(max_examples=40, deadline=None)
def test_all_enumerated_actions_are_legal(obs: Observation) -> None:
    legal = enumerate_legal_actions(obs)
    assert PASS_ACTION in legal
    for action in legal:
        assert is_legal_action(obs, action)
