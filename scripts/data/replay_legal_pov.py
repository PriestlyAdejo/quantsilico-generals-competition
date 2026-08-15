"""Legal fog-of-war POV reconstruction for elite replay snapshots.

ELITE_REPLAY_AUGMENTATION charter §4 HARD GATE: replay timelines contain FULL
state; no privileged information may reach policy inputs. This module turns a
full-state replay into per-player, per-tick LEGAL observations using
conservative generals.io visibility semantics:

- visible(t, p) = p's owned tiles + 4-neighbours of owned tiles (generals are
  owned, so general-vicinity vision is covered);
- terrain (mountains) and cities, once seen, remain known forever;
- tile ownership: current value while visible, last-seen value otherwise
  (the game keeps showing remembered ownership);
- army counts: own armies always known; enemy armies only while the tile is
  visible, otherwise the last-seen value is remembered WITHOUT refresh -
  consumers must treat stale army counts as memory, not truth;
- enemy generals: revealed only while visible (never through full state).

Pure numpy-free implementation (lists/sets) for auditability; inputs are the
raw payload dicts persisted by scripts/data/fetch_elite_replays.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

OWNED_EMPTY = -1
MOUNTAIN = -2
NEUTRAL_CITY = -3  # cities/castles treated as owned-by-none terrain features


@dataclass
class Replay:
    dims: tuple[int, int]
    players: list[str]
    mountains: set[tuple[int, int]]
    cities: set[tuple[int, int]]
    generals: dict[int, tuple[int, int]]  # player index -> initial general tile
    ticks: list[dict]  # each: {"armies": [[..]], "owners": [[..]]}
    winner: int | None
    seed: int | None = None


@dataclass
class PovMemory:
    """Per-player fog memory. Nothing here ever contains hidden live values."""

    seen_terrain: set[tuple[int, int]] = field(default_factory=set)
    seen_cities: set[tuple[int, int]] = field(default_factory=set)
    seen_generals: dict[int, tuple[int, int]] = field(default_factory=dict)
    last_owner: dict[tuple[int, int], int] = field(default_factory=dict)
    last_enemy_army: dict[tuple[int, int], int] = field(default_factory=dict)
    ingested: int = 0  # number of ticks already folded into memory


def parse_replay(payload: dict) -> Replay:
    raw_dims = payload["dims"]
    if isinstance(raw_dims, dict):
        dims = (int(raw_dims["rows"]), int(raw_dims["cols"]))
    else:
        dims = (int(raw_dims[0]), int(raw_dims[1]))
    mountains = {tuple(c) for c in payload.get("mountains", [])}
    cities = {tuple(c) for c in payload.get("castles", [])}
    generals = {i: tuple(g) for i, g in enumerate(payload.get("generals", []))}
    winner = payload.get("winner")
    return Replay(
        dims=dims,
        players=list(payload.get("players", [])),
        mountains=mountains,
        cities=cities,
        generals=generals,
        ticks=payload.get("ticks", []),
        winner=int(winner) if winner is not None else None,
        seed=payload.get("seed"),
    )


def visible_tiles(owners: list[list[int]], player: int) -> set[tuple[int, int]]:
    """Conservative legal vision: owned tiles plus their 4-neighbours."""
    owned = [
        (r, c) for r, row in enumerate(owners) for c, o in enumerate(row) if o == player
    ]
    vis = set(owned)
    for r, c in owned:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < len(owners) and 0 <= cc < len(owners[0]):
                vis.add((rr, cc))
    return vis


def legal_observation(
    replay: Replay, memory: PovMemory, player: int, tick_index: int
) -> dict:
    """Advance memory to tick_index and return the LEGAL view for player.

    Mutates memory (fog is stateful). The returned dict contains only what a
    real client could know; hidden enemy armies/generals are structurally
    absent rather than zeroed.
    """
    for t in range(memory.ingested, tick_index + 1):
        tick = replay.ticks[t]
        owners = tick["owners"]
        armies = tick["armies"]
        vis = visible_tiles(owners, player)
        for r, row in enumerate(owners):
            for c, owner in enumerate(row):
                pos = (r, c)
                if pos in replay.mountains or pos in replay.cities:
                    if pos in vis:
                        in_mountains = pos in replay.mountains
                        target = memory.seen_terrain if in_mountains else memory.seen_cities
                        target.add(pos)
                    continue
                if pos in vis:
                    memory.last_owner[pos] = owner
                    if owner != player and owner != OWNED_EMPTY and owner >= 0:
                        memory.last_enemy_army[pos] = armies[r][c]
                    if owner >= 0 and replay.generals.get(owner) == pos:
                        memory.seen_generals[owner] = pos
        memory.ingested = t + 1
    tick = replay.ticks[tick_index]
    owners = tick["owners"]
    armies = tick["armies"]
    vis = visible_tiles(owners, player)
    view = {
        "turn": tick_index,
        "visible": sorted(vis),
        "owner_view": {},
        "army_view": {},
        "generals_view": dict(memory.seen_generals),
        "mountains": sorted(memory.seen_terrain),
        "cities": sorted(memory.seen_cities),
    }
    for r, row in enumerate(owners):
        for c, owner in enumerate(row):
            pos = (r, c)
            if pos in replay.mountains or pos in replay.cities:
                continue
            remembered = memory.last_owner.get(pos)
            if remembered is None:
                continue  # never seen: structurally absent
            view["owner_view"][pos] = owner if pos in vis else remembered
            if owner == player:
                view["army_view"][pos] = armies[r][c]  # own armies always known
            elif pos in vis and owner >= 0:
                view["army_view"][pos] = armies[r][c]
            elif memory.last_enemy_army.get(pos) is not None:
                view["army_view"][pos] = memory.last_enemy_army[pos]  # stale memory
    return view


def memory_seen_count(memory: PovMemory) -> int:
    """Ticks already ingested into this memory."""
    return memory.ingested
