"""Smoke test: the private package must be importable."""

from __future__ import annotations

import generals_bot


def test_generals_bot_import() -> None:
    assert generals_bot.__version__ == "0.1.0"
