"""Belief package."""

from generals_bot.belief.opponent_posterior import (
    OpponentPosteriorState,
    TrajectoryFeatures,
    blend_posterior,
    enforce_unknown_floor,
    update_opponent_posterior,
)

__all__ = [
    "OpponentPosteriorState",
    "TrajectoryFeatures",
    "blend_posterior",
    "enforce_unknown_floor",
    "update_opponent_posterior",
]
