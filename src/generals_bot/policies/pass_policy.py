"""Permanent pass-bot fallback policy."""

from __future__ import annotations

from generals_bot.action import PASS_ACTION
from generals_bot.observation import GameContext, Observation
from generals_bot.policies.base import ActionDecision, PolicyState, TraceLevel


class PassPolicy:
    policy_id = "pass"

    def initial_state(self, context: GameContext) -> PolicyState:
        return PolicyState(data={"player_id": context.player_id})

    def act(
        self,
        observation: Observation,
        state: PolicyState,
        *,
        deterministic: bool,
        trace: TraceLevel,
        deadline: float | None,
    ) -> ActionDecision:
        return ActionDecision(
            action=PASS_ACTION,
            new_state=state,
            strategic_option="WAIT",
            option_distribution={"WAIT": 1.0},
            policy_id=self.policy_id,
            confidence=1.0,
            legal_action_count=1,
        )
