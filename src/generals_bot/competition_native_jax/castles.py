"""Castle price helpers for competition-native policy."""

from __future__ import annotations

import numpy as np

from generals_bot.castle_cost import castle_price_at, own_structures
from generals_bot.competition_native_jax.constants import MAX_HW
from generals_bot.observation import Observation


def price_examples() -> list[tuple[str, int, list[tuple[int, int]], tuple[int, int]]]:
    """Named fixtures: (name, expected_price, structures, target)."""
    return [
        ("no_nearby", 35, [(0, 0)], (10, 10)),
        ("d6", 37, [(0, 0)], (0, 6)),
        ("d5", 39, [(0, 0)], (0, 5)),
        ("d4", 41, [(0, 0)], (0, 4)),
        ("d3", 43, [(0, 0)], (0, 3)),
        ("d2", 45, [(0, 0)], (0, 2)),
        ("adjacent", 47, [(0, 0)], (0, 1)),
        ("two_d2", 55, [(0, 0), (0, 4)], (0, 2)),
    ]


def padded_price_map(observation: Observation) -> np.ndarray:
    """Return MAX_HW×MAX_HW price map; out-of-board cells get large sentinel."""
    structures = own_structures(observation)
    out = np.full((MAX_HW, MAX_HW), 10_000, dtype=np.int32)
    for r in range(observation.height):
        for c in range(observation.width):
            out[r, c] = castle_price_at(r, c, structures)
    return out
