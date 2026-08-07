"""Fix competition stdio loop to official multi-line protocol + student weights."""

from __future__ import annotations

import sys
from pathlib import Path

from generals_bot.agent import run_agent
from generals_bot.observation import GameContext
from generals_bot.policies.base import ActionDecision, PolicyState, TraceLevel
from generals_bot.competition_native_jax.policy import CompetitionNativePolicy, load_weights
from generals_bot.competition_native_jax.student_policy_numpy import StudentCompetitionNativePolicy


class _CNJAgentPolicy:
    policy_id = "competition_native_student"

    def __init__(self, inner: CompetitionNativePolicy) -> None:
        self.inner = inner

    def initial_state(self, context: GameContext) -> PolicyState:
        self.inner.reset(context.height, context.width)
        return PolicyState(data={"player_id": context.player_id})

    def act(self, observation, state, *, deterministic, trace, deadline):
        action, _ = self.inner.act(observation, deterministic=deterministic)
        return ActionDecision(
            action=action,
            new_state=state,
            policy_id=self.policy_id,
            strategic_option="LEARNED",
        )


def run_stdio(weights_path: Path | None = None, *, student: bool = True) -> None:
    """Official protocol via run_agent (no 'ready' line)."""
    weights = load_weights(weights_path) if weights_path else None
    if student:
        inner = StudentCompetitionNativePolicy(weights=weights, seed=0)
    else:
        inner = CompetitionNativePolicy(weights=weights, seed=0)
    run_agent(_CNJAgentPolicy(inner), deterministic=True)


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("weights.npz")
    run_stdio(path if path.exists() else None)
