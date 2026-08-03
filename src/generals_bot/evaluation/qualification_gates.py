"""Evaluate Phase 9Q three-level qualification gates from suite summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScreeningResult:
    passed: bool
    reasons: list[str]
    level: str = "SCREENING_SMOKE"


def evaluate_screening_smoke(
    *,
    expander: dict[str, Any],
    hunter: dict[str, Any],
    protocol_faults: int = 0,
    legal_action_rate: float = 1.0,
) -> ScreeningResult:
    """Screening gate: signal for A/B/C, not upload readiness."""
    reasons: list[str] = []
    wins = int(expander.get("wins", expander.get("W", 0)))
    losses = int(expander.get("losses", expander.get("L", 0)))
    discovery = float(expander.get("enemy_general_discovery_rate", expander.get("discovery_rate", 0.0)))
    h_wins = int(hunter.get("wins", hunter.get("W", 0)))
    h_losses = int(hunter.get("losses", hunter.get("L", 0)))
    h_games = int(hunter.get("games", h_wins + int(hunter.get("draws", hunter.get("D", 0))) + h_losses))

    if protocol_faults != 0:
        reasons.append(f"protocol_faults={protocol_faults}")
    if legal_action_rate < 1.0:
        reasons.append(f"legal_action_rate={legal_action_rate}")
    if losses > 1:
        reasons.append(f"expander_losses={losses}>1")
    if wins < 4:
        reasons.append(f"expander_wins={wins}<4")
    if discovery < 0.35:
        reasons.append(f"discovery={discovery:.3f}<0.35")
    if h_games >= 8 and h_wins == 0 and h_losses == 8:
        reasons.append("hunter=0/8")

    return ScreeningResult(passed=not reasons, reasons=reasons)


def evaluate_development_gate(
    *,
    groups: list[dict[str, Any]],
    discovery_rate: float,
    post_discovery_win_rate: float,
    hunter_comparable_to_v1: bool,
    protocol_faults: int = 0,
    legal_action_rate: float = 1.0,
) -> ScreeningResult:
    reasons: list[str] = []
    total_w = total_g = total_l = 0
    for g in groups:
        w = int(g.get("wins", 0))
        d = int(g.get("draws", 0))
        l = int(g.get("losses", 0))
        n = w + d + l
        total_w += w
        total_g += n
        total_l += l
        if n and (w / n) < 0.375:
            reasons.append(f"group_{g.get('name','?')}_win_rate={w/n:.3f}<0.375")
    if total_g == 0:
        reasons.append("no_games")
    else:
        if total_w / total_g < 0.50:
            reasons.append(f"combined_win_rate={total_w/total_g:.3f}<0.50")
        if total_l / total_g > 0.10:
            reasons.append(f"loss_rate={total_l/total_g:.3f}>0.10")
    if discovery_rate < 0.60:
        reasons.append(f"discovery={discovery_rate:.3f}<0.60")
    if post_discovery_win_rate < 0.80:
        reasons.append(f"post_discovery={post_discovery_win_rate:.3f}<0.80")
    if not hunter_comparable_to_v1:
        reasons.append("hunter_below_v1")
    if protocol_faults != 0:
        reasons.append(f"protocol_faults={protocol_faults}")
    if legal_action_rate < 1.0:
        reasons.append(f"legal_action_rate={legal_action_rate}")
    return ScreeningResult(passed=not reasons, reasons=reasons, level="DEVELOPMENT_GATE")


@dataclass(frozen=True)
class PrePpoSubmissionResult:
    passed: bool
    reasons: list[str]
    level: str = "PRE_PPO_SUBMISSION_GATE"


def evaluate_pre_ppo_submission_gate(
    *,
    paired_score_delta: float,
    paired_ci_low: float,
    protocol_faults: int,
    legal_action_rate: float,
    post_discovery_win_rate: float,
    conversion_micro_wins: int,
    conversion_micro_n: int,
    hunter_wins: int,
    hunter_losses: int,
    submitted_hunter_wins: int,
    submitted_hunter_losses: int,
    latency_p95_ms: float,
    peak_memory_mb: float | None,
    package_source_parity: bool,
) -> PrePpoSubmissionResult:
    """Second-submission gate vs the currently submitted portal package."""
    reasons: list[str] = []
    if protocol_faults != 0:
        reasons.append(f"protocol_faults={protocol_faults}")
    if legal_action_rate < 1.0:
        reasons.append(f"legal_action_rate={legal_action_rate}")
    if not package_source_parity:
        reasons.append("package_source_parity_failed")
    if paired_score_delta <= 0:
        reasons.append(f"paired_score_delta={paired_score_delta:.3f}<=0")
    if paired_ci_low < -0.08:
        reasons.append(f"paired_ci_low={paired_ci_low:.3f}<-0.08")
    if post_discovery_win_rate < 0.80:
        reasons.append(f"post_discovery={post_discovery_win_rate:.3f}<0.80")
    if conversion_micro_n and (conversion_micro_wins / conversion_micro_n) < 0.80:
        reasons.append(
            f"conversion_micro={conversion_micro_wins}/{conversion_micro_n}<0.80"
        )
    # No severe Hunter regression vs submitted package
    if hunter_wins + hunter_losses >= 4 and submitted_hunter_wins + submitted_hunter_losses >= 4:
        cand_score = (hunter_wins + 0.0) / max(1, hunter_wins + hunter_losses)
        sub_score = (submitted_hunter_wins + 0.0) / max(1, submitted_hunter_wins + submitted_hunter_losses)
        # Severe: wipeout relative to a submitted bot that already wins, or large win-rate collapse.
        if hunter_wins == 0 and submitted_hunter_wins >= 2 and hunter_losses >= 6:
            reasons.append("hunter_severe_regression_0_wins")
        if (sub_score - cand_score) >= 0.25 and hunter_losses - submitted_hunter_losses >= 2:
            reasons.append("hunter_loss_spike_vs_submitted")
    if latency_p95_ms >= 150.0:
        reasons.append(f"latency_p95_ms={latency_p95_ms:.1f}>=150")
    if peak_memory_mb is not None and peak_memory_mb >= 1800.0:
        reasons.append(f"peak_memory_mb={peak_memory_mb:.0f}>=1800")
    return PrePpoSubmissionResult(passed=not reasons, reasons=reasons)
