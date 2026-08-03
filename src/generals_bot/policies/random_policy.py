"""Seeded legal-random policy."""

from __future__ import annotations

import random

from generals_bot.legal import enumerate_legal_actions
from generals_bot.observation import GameContext, Observation
from generals_bot.policies.base import ActionDecision, PolicyState, TraceLevel


class RandomPolicy:
    policy_id = "legal_random"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def initial_state(self, context: GameContext) -> PolicyState:
        # Mix handshake identity into RNG so paired positions differ.
        rng_seed = (self.seed * 1_000_003 + context.player_id) & 0xFFFFFFFF
        return PolicyState(
            data={
                "player_id": context.player_id,
                "rng": random.Random(rng_seed),
                "turn_count": 0,
            }
        )

    def act(
        self,
        observation: Observation,
        state: PolicyState,
        *,
        deterministic: bool,
        trace: TraceLevel,
        deadline: float | None,
    ) -> ActionDecision:
        legal = enumerate_legal_actions(observation)
        rng: random.Random = state.data["rng"]
        if deterministic:
            # Stable pick: prefer moves over pass when present, else pass.
            action = legal[min(1, len(legal) - 1)] if len(legal) > 1 else legal[0]
        else:
            action = rng.choice(legal)
        state.data["turn_count"] = int(state.data.get("turn_count", 0)) + 1
        return ActionDecision(
            action=action,
            new_state=state,
            strategic_option="EXPAND",
            option_distribution={"EXPAND": 1.0},
            policy_id=self.policy_id,
            confidence=1.0 / max(len(legal), 1),
            legal_action_count=len(legal),
            top_candidates=legal[:5],
        )
