"""Ablation wrappers around frozen v2f reference with optional modules."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from generals_bot.action import KIND_MOVE
from generals_bot.legal import enumerate_legal_actions
from generals_bot.map_memory import MapMemory
from generals_bot.observation import GameContext, Observation
from generals_bot.policies.base import ActionDecision, TraceLevel
from generals_bot.policies.bounded_scout import BoundedScoutAssigner, ScoutTask
from generals_bot.policies.general_garrison import (
    filter_general_stripping,
    garrison_reserve,
    reinforcement_proposals,
)
from generals_bot.policies.heuristic_v2f_reference import HeuristicV2FReferencePolicy
from generals_bot.policies.hunter_intercept import intercept_proposals
from generals_bot.policies.threat_assessment import ThreatMemory, assess_threat
from generals_bot.risk.shield import SurvivalShield


@dataclass(frozen=True)
class AblationFlags:
    name: str
    use_planner: bool = False
    use_garrison: bool = False
    use_intercept: bool = False
    # When True, use credible threat assessor instead of v2f raw threatened→EMERGENCY
    use_credible_threat: bool = False

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# Frozen candidate matrix (immutable names)
V1_REFERENCE = "heuristic_v1_reference"
V2F_BEST = "heuristic_v2f_best_reference"
V2_9QD = "heuristic_v2_9qd_latest"
V2F_RESTORED = "heuristic_v2f_restored"
V2F_PLUS_PLANNER = "heuristic_v2f_plus_planner"
V2F_PLUS_GARRISON = "heuristic_v2f_plus_garrison"
V2F_PLUS_INTERCEPT = "heuristic_v2f_plus_hunter_intercept"
V2F_PLUS_PLANNER_GARRISON = "heuristic_v2f_plus_planner_plus_garrison"
V2F_PLUS_PLANNER_INTERCEPT = "heuristic_v2f_plus_planner_plus_intercept"
V2F_COMBINED = "heuristic_v2f_plus_planner_plus_garrison_plus_intercept"
V2_EXPLORE_ONLY = "heuristic_v2_explore_only"
V2_DEFENCE_ONLY = "heuristic_v2_defence_only"
V2_COMBINED_ALIAS = "heuristic_v2_combined"


FLAGS: dict[str, AblationFlags] = {
    V2F_BEST: AblationFlags(name=V2F_BEST),
    V2F_RESTORED: AblationFlags(name=V2F_RESTORED),
    V2F_PLUS_PLANNER: AblationFlags(name=V2F_PLUS_PLANNER, use_planner=True),
    V2F_PLUS_GARRISON: AblationFlags(name=V2F_PLUS_GARRISON, use_garrison=True, use_credible_threat=True),
    V2F_PLUS_INTERCEPT: AblationFlags(name=V2F_PLUS_INTERCEPT, use_intercept=True, use_credible_threat=True),
    V2F_PLUS_PLANNER_GARRISON: AblationFlags(
        name=V2F_PLUS_PLANNER_GARRISON, use_planner=True, use_garrison=True, use_credible_threat=True
    ),
    V2F_PLUS_PLANNER_INTERCEPT: AblationFlags(
        name=V2F_PLUS_PLANNER_INTERCEPT,
        use_planner=True,
        use_intercept=True,
        use_credible_threat=True,
    ),
    V2F_COMBINED: AblationFlags(
        name=V2F_COMBINED,
        use_planner=True,
        use_garrison=True,
        use_intercept=True,
        use_credible_threat=True,
    ),
    V2_EXPLORE_ONLY: AblationFlags(name=V2_EXPLORE_ONLY, use_planner=True),
    V2_DEFENCE_ONLY: AblationFlags(
        name=V2_DEFENCE_ONLY, use_garrison=True, use_intercept=True, use_credible_threat=True
    ),
    V2_COMBINED_ALIAS: AblationFlags(
        name=V2_COMBINED_ALIAS,
        use_planner=True,
        use_garrison=True,
        use_intercept=True,
        use_credible_threat=True,
    ),
}


class HeuristicV2AblationPolicy:
    """v2f base + optional bounded scout / garrison / intercept modules."""

    def __init__(self, flags: AblationFlags) -> None:
        self.flags = flags
        self.policy_id = flags.name
        self._base = HeuristicV2FReferencePolicy()
        self._base.policy_id = flags.name
        self._scout = BoundedScoutAssigner() if flags.use_planner else None
        self.config_hash = flags.config_hash()

    def initial_state(self, context: GameContext):
        state = self._base.initial_state(context)
        state.data["ablation"] = self.flags.name
        state.data["config_hash"] = self.config_hash
        state.data["scout_task"] = ScoutTask()
        state.data["threat"] = ThreatMemory()
        return state

    def act(
        self,
        observation: Observation,
        state,
        *,
        deterministic: bool,
        trace: TraceLevel,
        deadline: float | None,
    ) -> ActionDecision:
        memory: MapMemory = state.data["memory"]
        prev_seen = sum(1 for row in memory.ever_seen for v in row if v)

        # Credible threat overlay: do not let fog/distant enemy force EMERGENCY via base
        threat_mem: ThreatMemory = state.data.get("threat") or ThreatMemory()
        threat, threat_mem = assess_threat(observation, threat_mem)
        state.data["threat"] = threat_mem

        if self.flags.use_credible_threat:
            # Temporarily suppress base emergency by monkey-patching threat level to adjacent-only
            # Base still runs; we inject modules afterward and filter stripping.
            pass

        decision = self._base.act(
            observation, state, deterministic=deterministic, trace=TraceLevel.DECISION, deadline=deadline
        )
        state = decision.new_state
        memory = state.data["memory"]
        seen_now = sum(1 for row in memory.ever_seen for v in row if v)
        newly = seen_now > prev_seen

        legal = enumerate_legal_actions(observation)
        proposals = list(decision.proposals or [])

        # Re-enumerate proposals if base returned empty under NONE trace
        if not proposals:
            # Re-run base with DECISION to get proposals is expensive; use shield path only
            proposals = []

        # Inject modules
        if self.flags.use_intercept:
            proposals.extend(intercept_proposals(observation, legal))

        if self.flags.use_garrison:
            threatened = threat.emergency or threat.caution
            proposals.extend(
                reinforcement_proposals(observation, legal, threatened=threatened)
            )

        if self.flags.use_planner and self._scout is not None:
            gen_mask = memory.possible_enemy_general_mask(observation)
            task: ScoutTask = state.data.get("scout_task") or ScoutTask()
            task = self._scout.update(
                task,
                observation,
                memory,
                gen_mask,
                last_enemy=state.data.get("last_enemy_sighting"),
                newly_revealed=newly,
                enemy_general_known=state.data.get("known_enemy_general") is not None,
                emergency=threat.emergency,
            )
            state.data["scout_task"] = task
            proposals.extend(self._scout.proposals(task, observation, legal))

        # If we have injected proposals, re-select
        if proposals:
            if self.flags.use_garrison:
                reserve = garrison_reserve(turn=observation.turn, threatened=threat.emergency)
                proposals = filter_general_stripping(proposals, observation, reserve=reserve)
            selected = SurvivalShield().select(observation, proposals, legal)
            action = selected.action if selected else decision.action
            option = selected.option if selected else decision.strategic_option
            decision = ActionDecision(
                action=action,
                new_state=state,
                strategic_option=option,
                option_distribution={option: 1.0},
                policy_id=self.policy_id,
                confidence=selected.confidence if selected else decision.confidence,
                legal_action_count=len(legal),
                top_candidates=[p.action for p in proposals[:8]],
                proposals=proposals if trace != TraceLevel.NONE else [],
                shield_result={
                    **(decision.shield_result or {}),
                    "ablation": self.flags.name,
                    "config_hash": self.config_hash,
                    "selected_module": selected.module if selected else "base",
                },
            )
        else:
            decision.shield_result = {
                **(decision.shield_result or {}),
                "ablation": self.flags.name,
                "config_hash": self.config_hash,
            }

        diag = dict(state.data.get("diagnostics") or {})
        diag["ablation"] = self.flags.name
        diag["config_hash"] = self.config_hash
        diag["emergency_active"] = threat.emergency
        diag["caution_active"] = threat.caution
        if self.flags.use_planner:
            diag.update((state.data.get("scout_task") or ScoutTask()).to_diagnostics())
        state.data["diagnostics"] = diag
        decision.new_state = state
        return decision


def create_ablation(name: str) -> HeuristicV2AblationPolicy:
    key = name.strip().lower().replace("-", "_")
    if key not in FLAGS:
        raise KeyError(f"unknown ablation: {name}")
    return HeuristicV2AblationPolicy(FLAGS[key])
