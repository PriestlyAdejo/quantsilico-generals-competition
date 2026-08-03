"""Resumable autopilot campaign stages with resource guards."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STAGES = [
    "PRECHECK",
    "DATASET",
    "BC",
    "AUXILIARY",
    "PPO",
    "EVALUATION",
    "POPULATION",
    "EXPLAINABILITY",
    "PROMOTION",
    "PACKAGING",
    "LINUX_PARITY",
    "COMPLETE",
]


@dataclass
class StageState:
    name: str
    status: str = "PENDING"
    start_time: float | None = None
    end_time: float | None = None
    config: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    checkpoint: str | None = None
    resume_point: str | None = None
    error: str | None = None


@dataclass
class ResourceGuards:
    max_wall_clock_s: float = 7200.0
    max_env_steps: int = 50_000
    max_disk_gb: float = 20.0
    min_free_disk_gb: float = 5.0
    max_checkpoints: int = 20
    max_raw_replays: int = 50
    stop_requested: bool = False


@dataclass
class CampaignState:
    campaign_id: str
    stages: dict[str, StageState]
    guards: ResourceGuards
    started: float
    champion: str = "heuristic_v1"
    promotion_decision: str = "INSUFFICIENT_EVIDENCE"


def new_campaign(campaign_id: str = "bounded_initial") -> CampaignState:
    return CampaignState(
        campaign_id=campaign_id,
        stages={name: StageState(name=name) for name in STAGES},
        guards=ResourceGuards(max_wall_clock_s=7200.0),
        started=time.time(),
    )


def save_campaign(state: CampaignState, path: Path | None = None) -> Path:
    path = path or Path("experiments/manifests/autopilot_campaign.json")
    payload = {
        "campaign_id": state.campaign_id,
        "started": state.started,
        "champion": state.champion,
        "promotion_decision": state.promotion_decision,
        "guards": asdict(state.guards),
        "stages": {k: asdict(v) for k, v in state.stages.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_campaign(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mark_stage(state: CampaignState, name: str, status: str, **kwargs: Any) -> None:
    st = state.stages[name]
    if status == "RUNNING" and st.start_time is None:
        st.start_time = time.time()
    if status in {"PASS", "FAIL", "SKIPPED"}:
        st.end_time = time.time()
    st.status = status
    for k, v in kwargs.items():
        setattr(st, k, v)


def evaluate_promotion(
    *,
    champion: str = "heuristic_v1",
    challenger: str | None = None,
    bridge_decision: str,
    linux_parity: bool = False,
    package_windows: bool = True,
    empirical_payoff: bool = False,
    screen_passed: bool = False,
    holdout_passed: bool = False,
) -> dict[str, Any]:
    reasons: list[str] = []
    decision = "INSUFFICIENT_EVIDENCE"
    if challenger is None:
        reasons.append("no learned challenger nominated")
        decision = "NO LEARNED CANDIDATE PROMOTED"
    if bridge_decision == "FAIL":
        reasons.append("bridge FAIL blocks learned promotion")
        decision = "REJECTED_OPERATIONAL"
    if not empirical_payoff:
        reasons.append("empirical population evaluation incomplete for promotion")
    if challenger and not screen_passed:
        reasons.append("challenger failed or skipped development screen")
    if challenger and screen_passed and not holdout_passed:
        reasons.append("promotion holdout not passed")
        decision = "INSUFFICIENT_EVIDENCE"
    if not linux_parity:
        reasons.append("Linux parity not verified; cannot mark UPLOAD_READY for Windows-only package")
    upload_ready = bool(linux_parity and package_windows and champion == "heuristic_v1" and challenger is None)
    # Learned promotion only with full evidence
    if challenger and screen_passed and holdout_passed and empirical_payoff and linux_parity:
        decision = "PROMOTED"
        upload_ready = True
    elif challenger and decision == "INSUFFICIENT_EVIDENCE":
        pass
    elif challenger is None:
        decision = "NO LEARNED CANDIDATE PROMOTED"

    status = "UPLOAD_READY" if upload_ready and challenger is None else (
        "PACKAGED" if package_windows and champion == "heuristic_v1" else "RESEARCH"
    )
    if upload_ready and challenger is None:
        status = "UPLOAD_READY"
    report = {
        "schema_version": 1,
        "champion": champion,
        "challenger": challenger,
        "decision": decision,
        "champion_status": status,
        "UPLOAD_READY": upload_ready and challenger is None,
        "reasons": reasons,
        "bridge_decision": bridge_decision,
        "linux_parity": linux_parity,
        "empirical_payoff": empirical_payoff,
    }
    path = Path("experiments/manifests/promotion_decision.json")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
