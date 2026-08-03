"""Scoreboard residual helpers."""

from __future__ import annotations

from dataclasses import dataclass

from generals_bot.observation import Observation


@dataclass(frozen=True, slots=True)
class ScoreboardView:
    land_diff: int
    army_diff: int
    my_land: int
    my_army: int
    opp_land: int
    opp_army: int


def scoreboard_view(observation: Observation) -> ScoreboardView:
    return ScoreboardView(
        land_diff=observation.my_land - observation.opp_land,
        army_diff=observation.my_army - observation.opp_army,
        my_land=observation.my_land,
        my_army=observation.my_army,
        opp_land=observation.opp_land,
        opp_army=observation.opp_army,
    )
