"""Survival shield — hard safety overrides for general protection."""

from __future__ import annotations

from generals_bot.action import PASS_ACTION, Action
from generals_bot.legal import is_legal_action
from generals_bot.observation import Observation
from generals_bot.policies.base import Proposal


def proposal_rank_key(p: Proposal) -> tuple:
    """Deterministic total order for proposal selection.

    Primary (descending): hard_priority, score, confidence.
    Secondary (ascending): option, kind, row, col, direction, split.
    """
    a = p.action
    return (
        -int(p.hard_priority),
        -float(p.score),
        -float(p.confidence),
        str(p.option or ""),
        int(a.kind),
        int(a.row),
        int(a.col),
        int(a.direction),
        int(a.split),
    )


class SurvivalShield:
    """Select the highest hard_priority then score among legal proposals."""

    def select(
        self,
        observation: Observation,
        proposals: list[Proposal],
        legal: list[Action],
    ) -> Proposal | None:
        legal_set = {a.as_tuple() for a in legal}
        viable = [
            p
            for p in proposals
            if p.action.as_tuple() in legal_set
            and is_legal_action(observation, p.action)
            and not p.rejection_reasons
        ]
        if not viable:
            return Proposal(
                action=PASS_ACTION,
                option="WAIT",
                module="shield_fallback",
                hard_priority=0,
                score=0.0,
                confidence=1.0,
                explanation_code="shield_pass",
            )
        viable.sort(key=proposal_rank_key)
        return viable[0]
