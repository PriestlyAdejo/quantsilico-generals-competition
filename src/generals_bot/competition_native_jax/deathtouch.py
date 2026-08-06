"""Deathtouch / horizon feature helpers (no custom transitions)."""

from __future__ import annotations

import numpy as np

from generals_bot.competition_native_jax.constants import DEATHTOUCH_TURN, DRAW_TURN


def horizon_features(turn: int) -> dict[str, float]:
    t = float(turn)
    return {
        "x_turn": t / DRAW_TURN,
        "x_turn_mod50": (turn % 50) / 50.0,
        "x_death_countdown": float(np.clip((DEATHTOUCH_TURN - t) / DEATHTOUCH_TURN, -1.0, 1.0)),
        "x_dt_active": 1.0 if turn >= DEATHTOUCH_TURN else 0.0,
        "x_turns_since_dt": float(max(0, turn - DEATHTOUCH_TURN)) / DRAW_TURN,
        "x_cap_remaining": float(np.clip((DRAW_TURN - t) / DRAW_TURN, 0.0, 1.0)),
    }
