"""Castle cost tests aligned with official build_castles modifier."""

from __future__ import annotations

from generals_bot.castle_cost import castle_price_at


def test_base_price_far_from_structures() -> None:
    assert castle_price_at(10, 10, [(0, 0)]) == 35


def test_adjacent_surcharge() -> None:
    # d=1 → surcharge 12 → 47
    assert castle_price_at(0, 1, [(0, 0)]) == 47


def test_distance_table() -> None:
    general = [(0, 0)]
    expected = {
        0: 49,  # d=0: 14 → 49
        1: 47,
        2: 45,
        3: 43,
        4: 41,
        5: 39,
        6: 37,
        7: 35,
    }
    for distance, price in expected.items():
        assert castle_price_at(0, distance, general) == price


def test_stacked_surcharges() -> None:
    # Two structures at d=2 each: 35 + 10 + 10 = 55
    assert castle_price_at(2, 2, [(0, 2), (2, 0)]) == 55
