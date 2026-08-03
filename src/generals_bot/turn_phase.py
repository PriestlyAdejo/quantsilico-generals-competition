"""Turn-phase helpers for heuristic policies."""

from __future__ import annotations

from enum import StrEnum

from generals_bot.rules import DEATHTOUCH_TURN, DRAW_TURN, LAND_GROWTH_PERIOD


class TurnPhase(StrEnum):
    OPENING = "opening"
    MIDGAME = "midgame"
    DEATHTOUCH = "deathtouch"
    ENDGAME = "endgame"


def turn_phase(turn: int) -> TurnPhase:
    if turn < 50:
        return TurnPhase.OPENING
    if turn < DEATHTOUCH_TURN:
        return TurnPhase.MIDGAME
    if turn < DRAW_TURN - 100:
        return TurnPhase.DEATHTOUCH
    return TurnPhase.ENDGAME


def turns_to_land_growth(turn: int) -> int:
    mod = turn % LAND_GROWTH_PERIOD
    return 0 if mod == 0 else LAND_GROWTH_PERIOD - mod


def turns_to_deathtouch(turn: int) -> int:
    return max(0, DEATHTOUCH_TURN - turn)


def turns_to_draw(turn: int) -> int:
    return max(0, DRAW_TURN - turn)
