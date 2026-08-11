"""Canonical policy-visible belief memory (Phase 9F).

Wraps ``MapMemory`` so heuristic, specialists, ranker, PPO and deploy share one
lifecycle. Reset only on genuine episode terminal / handshake — never at
optimiser fragment boundaries.
"""

from __future__ import annotations

from generals_bot.map_memory import MapMemory
from generals_bot.observation import Observation

# Stable schema id for provenance hashing
BELIEF_SCHEMA_VERSION = "phase9f_belief_v1"


class BeliefMemory(MapMemory):
    """Alias with explicit Phase 9F semantics."""

    schema_version: str = BELIEF_SCHEMA_VERSION

    @classmethod
    def create(cls, height: int, width: int) -> BeliefMemory:
        base = MapMemory.create(height, width)
        return cls(
            height=base.height,
            width=base.width,
            known_terrain=base.known_terrain,
            last_owner=base.last_owner,
            last_army=base.last_army,
            info_age=base.info_age,
            ever_seen=base.ever_seen,
        )

    def update_visible(self, observation: Observation) -> None:
        self.update(observation)


def belief_channels_summary(belief: BeliefMemory) -> dict[str, float]:
    """Compact telemetry; not privileged hidden state."""
    h, w = belief.height, belief.width
    seen = sum(1 for r in range(h) for c in range(w) if belief.ever_seen[r][c])
    return {
        "coverage_frac": float(seen) / float(max(1, h * w)),
        "height": float(h),
        "width": float(w),
    }
