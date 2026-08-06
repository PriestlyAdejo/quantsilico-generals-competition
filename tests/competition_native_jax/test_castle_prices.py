"""Castle price unit tests."""

from __future__ import annotations

from generals_bot.castle_cost import castle_price_at
from generals_bot.competition_native_jax.castles import price_examples


def test_price_examples() -> None:
    for name, expected, structures, target in price_examples():
        got = castle_price_at(target[0], target[1], structures)
        assert got == expected, (name, got, expected)
