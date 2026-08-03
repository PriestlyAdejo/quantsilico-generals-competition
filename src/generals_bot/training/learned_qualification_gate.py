"""Learned-candidate qualification gate (pre-promotion)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LearnedQualificationGate:
    """Hard gates before a learned model may enter promotion evaluation."""

    official_venv_compatible: bool = False
    legal_action_rate: float = 0.0
    protocol_faults: int = -1
    safe_cpu_latency: bool = False
    qualification_development_completed: bool = False
    expander_wins: int = 0
    expander_draws: int = 0
    expander_losses: int = 0
    draw_conversion_metrics_present: bool = False
    general_defence_scenarios_completed: bool = False
    bc_val_exact_only: bool = False
    ppo_reward_only: bool = False
    population_score_rate_half: bool = False
    notes: list[str] = field(default_factory=list)

    def evaluate(self) -> dict[str, Any]:
        blockers: list[str] = []
        if not self.official_venv_compatible:
            blockers.append("official-.venv compatibility missing")
        if self.legal_action_rate < 1.0:
            blockers.append(f"legal_action_rate={self.legal_action_rate} < 1.0")
        if self.protocol_faults != 0:
            blockers.append(f"protocol_faults={self.protocol_faults} (require 0)")
        if not self.safe_cpu_latency:
            blockers.append("CPU latency not verified safe")
        if not self.qualification_development_completed:
            blockers.append("qualification development suite incomplete")
        if self.expander_losses > 0:
            blockers.append(f"Expander losses={self.expander_losses} (require 0 for gate)")
        if self.expander_wins + self.expander_draws + self.expander_losses <= 0:
            blockers.append("missing Expander W/D/L")
        if not self.draw_conversion_metrics_present:
            blockers.append("draw/conversion metrics missing")
        if not self.general_defence_scenarios_completed:
            blockers.append("general-defence scenarios incomplete")
        # Soft anti-patterns
        warnings: list[str] = []
        if self.bc_val_exact_only:
            warnings.append("BC validation accuracy alone is insufficient for promotion")
        if self.ppo_reward_only:
            warnings.append("PPO reward increase alone is insufficient for promotion")
        if self.population_score_rate_half:
            warnings.append("population score_rate≈0.5 may be persistent draws — require W/D/L")
        passed = len(blockers) == 0
        return {
            "schema_version": 1,
            "kind": "LEARNED_QUALIFICATION_GATE",
            "passed": passed,
            "blockers": blockers,
            "warnings": warnings,
            "expander_wdl": {
                "wins": self.expander_wins,
                "draws": self.expander_draws,
                "losses": self.expander_losses,
            },
            "inputs": asdict(self),
            "note": "Do not promote on BC accuracy, PPO reward, or score_rate alone.",
        }
