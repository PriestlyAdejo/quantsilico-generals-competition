"""Promotion policy (EXECUTION_PLAN section 7.3, configs/marathon/programme.yaml)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    pathway: str  # NORMAL | ROBUSTNESS | NO_PROMOTION
    reason: str
    lower_bound: float
    practical_margin: float


def decide_promotion(
    *,
    lower_bound: float,
    practical_margin: float = 0.01,
    robustness_lower: float | None = None,
    robustness_noninferiority_margin: float = -0.005,
    worst_matchup_improvement: float | None = None,
    worst_matchup_threshold: float = 0.05,
    integrity_latency_fault_gates_pass: bool = True,
) -> PromotionDecision:
    """Normal promotion requires the CS lower bound above the practical margin.

    Statistical evidence above zero alone is insufficient.  A documented
    robustness promotion is permitted when aggregate performance is no worse
    than the configured noninferiority margin and the known worst matchup
    improves by at least the configured threshold, with the same integrity
    gates.
    """
    if not integrity_latency_fault_gates_pass:
        return PromotionDecision(
            promoted=False,
            pathway="NO_PROMOTION",
            reason="integrity/latency/fault gates failed",
            lower_bound=lower_bound,
            practical_margin=practical_margin,
        )
    if lower_bound > practical_margin:
        return PromotionDecision(
            promoted=True,
            pathway="NORMAL",
            reason=(
                f"CS lower bound {lower_bound:.5f} exceeds practical margin "
                f"{practical_margin:.5f}"
            ),
            lower_bound=lower_bound,
            practical_margin=practical_margin,
        )
    if (
        robustness_lower is not None
        and worst_matchup_improvement is not None
        and robustness_lower >= robustness_noninferiority_margin
        and worst_matchup_improvement >= worst_matchup_threshold
    ):
        return PromotionDecision(
            promoted=True,
            pathway="ROBUSTNESS",
            reason=(
                f"noninferiority lower bound {robustness_lower:.5f} >= "
                f"{robustness_noninferiority_margin:.5f} with worst matchup improvement "
                f"{worst_matchup_improvement:.5f} >= {worst_matchup_threshold:.5f}"
            ),
            lower_bound=lower_bound,
            practical_margin=practical_margin,
        )
    return PromotionDecision(
        promoted=False,
        pathway="NO_PROMOTION",
        reason=(
            f"CS lower bound {lower_bound:.5f} does not exceed practical margin "
            f"{practical_margin:.5f} and robustness criteria are not met"
        ),
        lower_bound=lower_bound,
        practical_margin=practical_margin,
    )
