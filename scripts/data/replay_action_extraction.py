"""Action extraction from full-state elite replays (dependency-safe milestone).

Depends only on the replay payload schema (see replay_legal_pov.parse_replay).
Between tick t and t+1, each player's move is classified as MOVE / BUILD /
PASS with the moved-army estimate and a legality flag against that player's
fog (legal-POV hard gate): a MOVE is only legal-sampled if the source tile is
owned by the player. Amounts are ESTIMATES - tick granularity and recruitment
mean exact split fractions are not recoverable from public replays; consumers
must treat extracted actions as behavioural targets, not ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActionEvent:
    tick: int
    player: int
    kind: str  # MOVE | BUILD | PASS
    src: tuple[int, int] | None = None
    dst: tuple[int, int] | None = None
    amount: int = 0
    legal_pov: bool = True


def _owned(owners: list[list[int]], player: int) -> set[tuple[int, int]]:
    return {
        (r, c) for r, row in enumerate(owners) for c, o in enumerate(row) if o == player
    }


def _adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def extract_tick_actions(
    owners_prev: list[list[int]],
    armies_prev: list[list[int]],
    owners_next: list[list[int]],
    armies_next: list[list[int]],
    cities: set[tuple[int, int]],
    player: int,
    tick: int,
) -> list[ActionEvent]:
    """Classify player's action between two consecutive ticks.

    Heuristic v1: find the largest army decrease on player-owned tiles (source)
    and the largest army/ownership increase elsewhere (destination); if both
    exist and are adjacent it is a MOVE. A city tile that becomes owned is a
    BUILD. No detectable change is a PASS. Legality is checked against the
    player's own ownership only (never against hidden enemy state).
    """
    owned_prev = _owned(owners_prev, player)
    owned_next = _owned(owners_next, player)
    events: list[ActionEvent] = []

    new_cities = [
        pos for pos in owned_next & cities if pos not in owned_prev
    ]
    for pos in new_cities:
        events.append(ActionEvent(tick, player, "BUILD", dst=pos, legal_pov=True))

    best_src, best_drop = None, 0
    for pos in owned_prev:
        drop = armies_prev[pos[0]][pos[1]] - armies_next[pos[0]][pos[1]]
        if drop > best_drop:
            best_src, best_drop = pos, drop
    best_dst, best_gain = None, 0
    for r, row in enumerate(owners_next):
        for c, owner in enumerate(row):
            pos = (r, c)
            if pos in cities or pos in owned_prev:
                continue
            gain = armies_next[r][c] - (armies_prev[r][c] if owners_prev[r][c] == player else 0)
            if owner == player and gain > best_gain:
                best_dst, best_gain = pos, gain
    if best_src is not None and best_dst is not None and best_drop >= 1 and best_gain >= 1:
        legal = _adjacent(best_src, best_dst)
        events.append(
            ActionEvent(
                tick,
                player,
                "MOVE",
                src=best_src,
                dst=best_dst,
                amount=min(best_drop, best_gain),
                legal_pov=legal,
            )
        )
        return events
    if not events:
        events.append(ActionEvent(tick, player, "PASS", legal_pov=True))
    return events


def extract_player_actions(replay, player: int) -> list[ActionEvent]:
    """Full-replay action timeline for one player (full-state extraction;
    downstream BC sampling must filter through replay_legal_pov views)."""
    out: list[ActionEvent] = []
    for t in range(len(replay.ticks) - 1):
        prev, nxt = replay.ticks[t], replay.ticks[t + 1]
        out.extend(
            extract_tick_actions(
                prev["owners"], prev["armies"], nxt["owners"], nxt["armies"],
                replay.cities, player, t,
            )
        )
    return out
