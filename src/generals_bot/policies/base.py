"""Common policy interface for heuristic and learned candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from generals_bot.action import PASS_ACTION, Action
from generals_bot.observation import GameContext, Observation


class TraceLevel(StrEnum):
    NONE = "none"
    MINIMAL = "minimal"
    DECISION = "decision"
    FULL_OFFLINE = "full_offline"


@dataclass
class PolicyState:
    """Opaque per-match policy state; reset at handshake."""

    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Proposal:
    action: Action
    option: str
    module: str
    hard_priority: int
    score: float
    confidence: float
    explanation_code: str
    explanation_values: dict[str, float] = field(default_factory=dict)
    rejection_reasons: tuple[str, ...] = ()


@dataclass
class ActionDecision:
    action: Action
    new_state: PolicyState
    strategic_option: str = "WAIT"
    option_distribution: dict[str, float] = field(default_factory=dict)
    policy_id: str = ""
    model_id: str | None = None
    checkpoint_hash: str | None = None
    value: float | None = None
    general_loss_risk: float | None = None
    confidence: float = 1.0
    legal_action_count: int = 0
    top_candidates: list[Action] = field(default_factory=list)
    opponent_posterior: dict[str, float] = field(default_factory=dict)
    search_result: dict[str, Any] | None = None
    shield_result: dict[str, Any] | None = None
    trace_reference: str | None = None
    latency_ms: float | None = None
    fallback_used: bool = False
    proposals: list[Proposal] = field(default_factory=list)


class Policy(Protocol):
    policy_id: str

    def initial_state(self, context: GameContext) -> PolicyState:
        ...

    def act(
        self,
        observation: Observation,
        state: PolicyState,
        *,
        deterministic: bool,
        trace: TraceLevel,
        deadline: float | None,
    ) -> ActionDecision:
        ...


def fallback_pass_decision(state: PolicyState, *, policy_id: str) -> ActionDecision:
    return ActionDecision(
        action=PASS_ACTION,
        new_state=state,
        strategic_option="WAIT",
        policy_id=policy_id,
        fallback_used=True,
        legal_action_count=1,
    )
