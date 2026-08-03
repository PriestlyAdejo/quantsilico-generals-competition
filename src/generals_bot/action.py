"""Action types for competition agents."""

from __future__ import annotations

from dataclasses import dataclass

KIND_MOVE = 0
KIND_PASS = 1
KIND_BUILD = 2


@dataclass(frozen=True, slots=True)
class Action:
    """One protocol action: move, pass, or build."""

    kind: int
    row: int = 0
    col: int = 0
    direction: int = 0
    split: int = 0

    def as_tuple(self) -> tuple[int, int, int, int, int]:
        return (self.kind, self.row, self.col, self.direction, self.split)

    @classmethod
    def pass_(cls) -> Action:
        return PASS_ACTION

    @classmethod
    def move(cls, row: int, col: int, direction: int, *, split: int = 0) -> Action:
        return cls(kind=KIND_MOVE, row=row, col=col, direction=direction, split=split)

    @classmethod
    def build(cls, row: int, col: int) -> Action:
        return cls(kind=KIND_BUILD, row=row, col=col, direction=0, split=0)


PASS_ACTION = Action(kind=KIND_PASS, row=0, col=0, direction=0, split=0)
