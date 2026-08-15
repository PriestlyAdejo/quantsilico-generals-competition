"""Canonical pair construction and scoring (EXECUTION_PLAN section 7.1)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class PairSchedule:
    """One seat-swapped pair: same map seed, candidate plays both seats."""

    pair_id: str
    opponent_id: str
    map_seed: int
    game_seats: tuple[tuple[str, int], ...]  # (candidate_seat, game_index)

    @property
    def game_count(self) -> int:
        return len(self.game_seats)


def canonical_map_seed(eval_namespace: str, opponent_id: str, pair_index: int) -> int:
    """Deterministic map seed disjoint from training/screening namespaces.

    Seeds derive from a stable hash so candidate/incumbent/evaluator all
    reconstruct identical maps without shared mutable state.
    """
    material = f"{eval_namespace}|{opponent_id}|pair|{pair_index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31 - 1)


def pair_schedule(
    *,
    eval_namespace: str,
    opponent_id: str,
    pair_count: int,
    start_index: int = 0,
) -> list[PairSchedule]:
    if pair_count < 0:
        raise ValueError(f"pair_count must be non-negative: {pair_count}")
    schedule = []
    for offset in range(pair_count):
        index = start_index + offset
        schedule.append(
            PairSchedule(
                pair_id=f"{opponent_id}#{index}",
                opponent_id=opponent_id,
                map_seed=canonical_map_seed(eval_namespace, opponent_id, index),
                game_seats=(("A", 0), ("B", 1)),
            )
        )
    return schedule


def pair_score_from_game_scores(
    candidate_seat_a_score: float, candidate_seat_b_score: float
) -> float:
    """PAIR_SCORE = mean candidate score across the two seat-swapped games."""
    return (candidate_seat_a_score + candidate_seat_b_score) / 2.0


def game_score(outcome: str, *, win: float = 1.0, draw: float = 0.5, loss: float = 0.0) -> float:
    """Canonical per-game score unless programme.yaml explicitly supersedes."""
    normalized = outcome.strip().upper()
    if normalized == "WIN":
        return win
    if normalized == "DRAW":
        return draw
    if normalized == "LOSS":
        return loss
    raise ValueError(f"unknown game outcome: {outcome!r}")
