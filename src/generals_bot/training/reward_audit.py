"""PPO / shaping reward audit for qualification-hardening.

Draws must not appear equivalent to wins under land/army shaping.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TerminalRewardConfig:
    """Explicit terminal outcomes. Prefer ablation before locking draw_penalty."""

    win: float = 1.0
    draw: float = 0.0
    loss: float = -1.0
    name: str = "win1_draw0_loss-1"

    def terminal_reward(self, *, winner: int | None, perspective: int = 0) -> float:
        if winner is None or winner < 0:
            return self.draw
        if winner == perspective:
            return self.win
        return self.loss


DRAW_NEUTRAL = TerminalRewardConfig(win=1.0, draw=0.0, loss=-1.0, name="win1_draw0_loss-1")
DRAW_PENALTY = TerminalRewardConfig(win=1.0, draw=-0.2, loss=-1.0, name="win1_draw-0.2_loss-1")

ABLATION_CONFIGS = (DRAW_NEUTRAL, DRAW_PENALTY)


@dataclass
class ShapingAuditFinding:
    signal: str
    can_be_positive_on_draw: bool
    risk: str
    recommendation: str


def audit_shaping_signals() -> list[ShapingAuditFinding]:
    """Static audit of known shaping channels that can reward territorial draws."""
    return [
        ShapingAuditFinding(
            signal="land_dominance_delta",
            can_be_positive_on_draw=True,
            risk="Turn-1200 territorial dominance can accumulate positive shaping without a win.",
            recommendation="Zero or decay land shaping after turn 800; never treat draw as success.",
        ),
        ShapingAuditFinding(
            signal="army_dominance_delta",
            can_be_positive_on_draw=True,
            risk="Army lead at draw looks like success under dense shaping.",
            recommendation="Cap army shaping; emphasise general-capture sparse bonus.",
        ),
        ShapingAuditFinding(
            signal="castle_ownership",
            can_be_positive_on_draw=True,
            risk="Castles produce ongoing army that can raise draw shaping without conversion.",
            recommendation="Do not reward castle count alone after conversion phase.",
        ),
        ShapingAuditFinding(
            signal="terminal_draw_as_zero",
            can_be_positive_on_draw=False,
            risk="draw=0 makes Expander draw loops look as good as mixed W/L at score_rate 0.5.",
            recommendation="Ablate DRAW_PENALTY (-0.2) vs DRAW_NEUTRAL; require W/D/L promotion metrics.",
        ),
    ]


def reward_audit_report() -> dict[str, Any]:
    findings = audit_shaping_signals()
    return {
        "schema_version": 1,
        "kind": "PPO_REWARD_AUDIT",
        "status": "IMPLEMENTED_CONFIG_NOT_CAMPAIGN",
        "terminal_ablation": [asdict(c) for c in ABLATION_CONFIGS],
        "default_for_smoke": asdict(DRAW_PENALTY),
        "findings": [asdict(f) for f in findings],
        "promotion_rule": (
            "Do not promote on score_rate or mean reward alone; require Expander W/D/L "
            "and dominant-position conversion."
        ),
        "campaign_gate": "120-minute PPO campaign blocked until qualification-hardening passes.",
    }
