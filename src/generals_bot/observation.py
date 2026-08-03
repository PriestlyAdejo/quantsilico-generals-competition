"""Observation and related view types for competition agents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Observation:
    """One partial-observation frame from the stdio protocol."""

    height: int
    width: int
    turn: int
    my_land: int
    my_army: int
    opp_land: int
    opp_army: int
    type_grid: tuple[tuple[int, ...], ...]
    owner_grid: tuple[tuple[int, ...], ...]
    army_grid: tuple[tuple[int, ...], ...]

    @property
    def H(self) -> int:  # noqa: N802 - match official starter naming
        return self.height

    @property
    def W(self) -> int:  # noqa: N802 - match official starter naming
        return self.width


@dataclass(frozen=True, slots=True)
class GameContext:
    """Static match context from the handshake."""

    player_id: int
    height: int
    width: int
