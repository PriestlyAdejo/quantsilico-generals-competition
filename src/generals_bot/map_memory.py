"""Persistent map memory within a single match (reset at handshake)."""

from __future__ import annotations

from dataclasses import dataclass, field

from generals_bot.observation import Observation
from generals_bot.protocol import (
    OWNER_ME,
    OWNER_OPP,
    TYPE_FOG,
    TYPE_MOUNTAIN,
    TYPE_STRUCTURE_IN_FOG,
)


@dataclass
class MapMemory:
    height: int
    width: int
    known_terrain: list[list[int | None]] = field(default_factory=list)
    last_owner: list[list[int | None]] = field(default_factory=list)
    last_army: list[list[int | None]] = field(default_factory=list)
    info_age: list[list[int]] = field(default_factory=list)
    ever_seen: list[list[bool]] = field(default_factory=list)

    @classmethod
    def create(cls, height: int, width: int) -> MapMemory:
        return cls(
            height=height,
            width=width,
            known_terrain=[[None] * width for _ in range(height)],
            last_owner=[[None] * width for _ in range(height)],
            last_army=[[None] * width for _ in range(height)],
            info_age=[[10**9] * width for _ in range(height)],
            ever_seen=[[False] * width for _ in range(height)],
        )

    def update(self, observation: Observation) -> None:
        for r in range(self.height):
            for c in range(self.width):
                cell_type = observation.type_grid[r][c]
                if cell_type == TYPE_FOG:
                    self.info_age[r][c] += 1
                    continue
                self.ever_seen[r][c] = True
                self.info_age[r][c] = 0
                if cell_type not in (TYPE_STRUCTURE_IN_FOG,):
                    self.known_terrain[r][c] = cell_type
                elif self.known_terrain[r][c] is None:
                    self.known_terrain[r][c] = TYPE_MOUNTAIN  # unknown structure
                self.last_owner[r][c] = observation.owner_grid[r][c]
                self.last_army[r][c] = observation.army_grid[r][c]

    def possible_enemy_general_mask(self, observation: Observation) -> list[list[bool]]:
        """Cells that could still hide the enemy general."""
        mask = [[False] * self.width for _ in range(self.height)]
        for r in range(self.height):
            for c in range(self.width):
                terrain = self.known_terrain[r][c]
                if terrain == TYPE_MOUNTAIN:
                    continue
                owner_now = observation.owner_grid[r][c]
                if owner_now == OWNER_ME:
                    continue
                # Visible enemy general already found
                if observation.type_grid[r][c] == 4 and owner_now == OWNER_OPP:
                    mask[r][c] = True
                    continue
                if observation.type_grid[r][c] == TYPE_FOG or not self.ever_seen[r][c]:
                    mask[r][c] = True
        return mask
