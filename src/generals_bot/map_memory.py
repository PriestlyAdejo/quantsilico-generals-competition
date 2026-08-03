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

    def possible_enemy_general_mask(
        self,
        observation: Observation,
        *,
        min_general_distance: int = 17,
    ) -> list[list[bool]]:
        """Cells that could still hide the enemy general (no privileged hidden state)."""
        from generals_bot.protocol import TYPE_GENERAL

        own_gen: tuple[int, int] | None = None
        enemy_cells: list[tuple[int, int]] = []
        for r in range(self.height):
            for c in range(self.width):
                if (
                    observation.owner_grid[r][c] == OWNER_ME
                    and observation.type_grid[r][c] == TYPE_GENERAL
                ):
                    own_gen = (r, c)
                if observation.owner_grid[r][c] == OWNER_OPP:
                    enemy_cells.append((r, c))
                    if observation.type_grid[r][c] == TYPE_GENERAL:
                        mask = [[False] * self.width for _ in range(self.height)]
                        mask[r][c] = True
                        return mask
                # Remember last-seen enemy positions from memory
                if self.last_owner[r][c] == OWNER_OPP:
                    enemy_cells.append((r, c))

        mask = [[False] * self.width for _ in range(self.height)]
        for r in range(self.height):
            for c in range(self.width):
                if self.known_terrain[r][c] == TYPE_MOUNTAIN:
                    continue
                if observation.owner_grid[r][c] == OWNER_ME:
                    continue
                cell_type = observation.type_grid[r][c]
                # Only unresolved fog (or never seen) can still hide a general
                if cell_type == TYPE_STRUCTURE_IN_FOG:
                    continue
                if cell_type != TYPE_FOG and self.ever_seen[r][c]:
                    continue
                if own_gen is not None:
                    if abs(r - own_gen[0]) + abs(c - own_gen[1]) < min_general_distance:
                        continue
                if enemy_cells:
                    if min(abs(r - er) + abs(c - ec) for er, ec in enemy_cells) > 10:
                        # Still allow far fog if never contacted deeply — keep a thinner set
                        if cell_type == TYPE_FOG and not self._adjacent_to_owned(observation, r, c):
                            continue
                mask[r][c] = True
        return mask

    def _adjacent_to_owned(self, observation: Observation, r: int, c: int) -> bool:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                if observation.owner_grid[nr][nc] == OWNER_ME:
                    return True
        return False

    def enemy_frontier_fog_targets(self, observation: Observation) -> list[tuple[int, int, int]]:
        """Return (row, col, priority) fog cells adjacent to us or to visible enemy."""
        targets: list[tuple[int, int, int]] = []
        for r in range(self.height):
            for c in range(self.width):
                if observation.type_grid[r][c] != TYPE_FOG:
                    continue
                pri = 0
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < self.height and 0 <= nc < self.width):
                        continue
                    if observation.owner_grid[nr][nc] == OWNER_OPP:
                        pri = max(pri, 100)
                    elif observation.owner_grid[nr][nc] == OWNER_ME:
                        pri = max(pri, 40)
                if pri:
                    pri += min(50, self.info_age[r][c] // 5)
                    if not self.ever_seen[r][c]:
                        pri += 20
                    targets.append((r, c, pri))
        targets.sort(key=lambda x: -x[2])
        return targets
