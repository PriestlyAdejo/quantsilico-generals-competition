"""Ablation wrappers around frozen v2f reference with optional modules."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from generals_bot.legal import enumerate_legal_actions
from generals_bot.map_memory import MapMemory
from generals_bot.observation import GameContext
from generals_bot.policies.base import ActionDecision, TraceLevel
from generals_bot.policies.bounded_scout import BoundedScoutAssigner, ScoutTask
from generals_bot.policies.exploration_planner import ExplorationState
from generals_bot.policies.general_garrison import (
    filter_general_stripping,
    garrison_reserve,
    reinforcement_proposals,
)
from generals_bot.policies.general_hunt_plan import (
    GeneralHuntPlan,
    hunt_plan_proposals,
    update_hunt_plan,
)
from generals_bot.policies.heuristic_v2f_reference import HeuristicV2FReferencePolicy
from generals_bot.policies.hunter_intercept import corridor_intercept_proposals_v2, intercept_proposals
from generals_bot.policies.terminal_win_oracle import immediate_terminal_win_proposals
from generals_bot.policies.threat_assessment import ThreatMemory, assess_threat
from generals_bot.risk.shield import SurvivalShield


@dataclass(frozen=True)
class AblationFlags:
    name: str
    use_planner: bool = False
    use_garrison: bool = False
    use_intercept: bool = False
    use_intercept_v2: bool = False
    use_terminal_oracle: bool = False
    use_hunt_plan: bool = False
    # When True, use credible threat assessor instead of v2f raw threatened→EMERGENCY
    use_credible_threat: bool = False
    use_persistent_explore: bool = False
    use_dual_scout: bool = False

    def config_hash(self) -> str:
        payload_obj = asdict(self)
        # Preserve historical hashes for candidates that do not use newer flags.
        for optional in (
            "use_intercept_v2",
            "use_terminal_oracle",
            "use_hunt_plan",
            "use_persistent_explore",
            "use_dual_scout",
        ):
            if not payload_obj.get(optional):
                payload_obj.pop(optional, None)
        payload = json.dumps(payload_obj, sort_keys=True)
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
V2F_PLANNER_CORRIDOR_V2 = "heuristic_v2f_planner_plus_corridor_intercept_v2"
V2F_PLANNER_TERMINAL = "heuristic_v2f_plus_planner_terminal_fix"
V2F_CORRIDOR_TERMINAL = "heuristic_v2f_planner_plus_corridor_v2_terminal_fix"
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
    V2F_PLANNER_CORRIDOR_V2: AblationFlags(
        name=V2F_PLANNER_CORRIDOR_V2,
        use_planner=True,
        use_intercept_v2=True,
        use_credible_threat=True,
    ),
    V2F_PLANNER_TERMINAL: AblationFlags(
        name=V2F_PLANNER_TERMINAL,
        use_planner=True,
        use_terminal_oracle=True,
        use_hunt_plan=True,
        use_credible_threat=True,
        use_persistent_explore=True,
        use_dual_scout=True,
    ),
    V2F_CORRIDOR_TERMINAL: AblationFlags(
        name=V2F_CORRIDOR_TERMINAL,
        use_planner=True,
        use_intercept_v2=True,
        use_terminal_oracle=True,
        use_hunt_plan=True,
        use_credible_threat=True,
        use_persistent_explore=True,
        use_dual_scout=True,
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
    """v2f base + optional bounded scout / garrison / intercept / terminal modules."""

    def __init__(self, flags: AblationFlags) -> None:
        self.flags = flags
        self.policy_id = flags.name
        self._base = HeuristicV2FReferencePolicy()
        self._base.policy_id = flags.name
        self._scout = (
            BoundedScoutAssigner(dual_scout=flags.use_dual_scout) if flags.use_planner else None
        )
        self.config_hash = flags.config_hash()

    def initial_state(self, context: GameContext):
        state = self._base.initial_state(context)
        state.data["ablation"] = self.flags.name
        state.data["config_hash"] = self.config_hash
        state.data["scout_task"] = ScoutTask()
        state.data["scout_task_b"] = ScoutTask()
        state.data["exploration_state"] = ExplorationState()
        state.data["threat"] = ThreatMemory()
        state.data["hunt_plan"] = GeneralHuntPlan()
        return state

    def act(
        self,
        observation,
        state,
        *,
        deterministic: bool,
        trace: TraceLevel,
        deadline: float | None,
    ) -> ActionDecision:
        memory: MapMemory = state.data["memory"]
        prev_seen = sum(1 for row in memory.ever_seen for v in row if v)

        threat_mem: ThreatMemory = state.data.get("threat") or ThreatMemory()
        threat, threat_mem = assess_threat(observation, threat_mem)
        state.data["threat"] = threat_mem

        decision = self._base.act(
            observation, state, deterministic=deterministic, trace=TraceLevel.DECISION, deadline=deadline
        )
        state = decision.new_state
        memory = state.data["memory"]
        seen_now = sum(1 for row in memory.ever_seen for v in row if v)
        newly = seen_now > prev_seen

        legal = enumerate_legal_actions(observation)
        proposals = list(decision.proposals or [])
        known_eg = state.data.get("known_enemy_general")
        # Only exact adjacent visible threat should interrupt hunt for terminal wins
        emergency_for_hunt = bool(threat.emergency)

        # Inject modules
        if self.flags.use_intercept:
            proposals.extend(intercept_proposals(observation, legal))
        if self.flags.use_intercept_v2:
            proposals.extend(corridor_intercept_proposals_v2(observation, legal))

        if self.flags.use_garrison:
            threatened = threat.emergency or threat.caution
            proposals.extend(reinforcement_proposals(observation, legal, threatened=threatened))

        if self.flags.use_planner and self._scout is not None:
            gen_mask = memory.possible_enemy_general_mask(observation)
            task: ScoutTask = state.data.get("scout_task") or ScoutTask()
            task_b: ScoutTask = state.data.get("scout_task_b") or ScoutTask()
            est: ExplorationState = state.data.get("exploration_state") or ExplorationState()
            # Persistent ExplorationState lives on policy state across turns.
            task, task_b, est = self._scout.update(
                task,
                observation,
                memory,
                gen_mask,
                last_enemy=state.data.get("last_enemy_sighting"),
                newly_revealed=newly,
                enemy_general_known=known_eg is not None,
                emergency=threat.emergency,
                exploration=est if self.flags.use_persistent_explore else ExplorationState(
                    last_reveal_turn=task.last_reveal_turn,
                    stalled_turns=task.stall,
                ),
                secondary=task_b if self.flags.use_dual_scout else ScoutTask(),
            )
            state.data["scout_task"] = task
            if self.flags.use_dual_scout:
                state.data["scout_task_b"] = task_b
            if self.flags.use_persistent_explore:
                state.data["exploration_state"] = est
            if known_eg is None:
                proposals.extend(self._scout.proposals(task, observation, legal))
                if self.flags.use_dual_scout:
                    proposals.extend(self._scout.proposals(task_b, observation, legal))

        if self.flags.use_hunt_plan and known_eg is not None:
            plan: GeneralHuntPlan = state.data.get("hunt_plan") or GeneralHuntPlan()
            plan = update_hunt_plan(
                plan,
                observation,
                known_general=known_eg,
                emergency=emergency_for_hunt,
            )
            state.data["hunt_plan"] = plan
            proposals.extend(hunt_plan_proposals(plan, observation, legal))
            # Once hunting, do not let ordinary expand/castle/false defence starve the route.
            if not emergency_for_hunt:
                filtered = []
                for p in proposals:
                    if p.option == "IMMEDIATE_TERMINAL_WIN":
                        filtered.append(p)
                        continue
                    if p.option in {"EXPAND", "CASTLE", "SCOUT"} and p.hard_priority < 100:
                        continue
                    if p.option == "DEFEND" and p.hard_priority <= 100:
                        continue
                    filtered.append(p)
                proposals = filtered

        terminal_hits = []
        if self.flags.use_terminal_oracle:
            terminal_hits = immediate_terminal_win_proposals(
                observation, known_eg, legal=legal, hard_priority=110
            )
            proposals.extend(terminal_hits)

        if proposals:
            if self.flags.use_garrison:
                reserve = garrison_reserve(turn=observation.turn, threatened=threat.emergency)
                # Never strip-filter verified terminal wins
                kept_terminal = [p for p in proposals if p.option == "IMMEDIATE_TERMINAL_WIN"]
                other = [p for p in proposals if p.option != "IMMEDIATE_TERMINAL_WIN"]
                other = filter_general_stripping(other, observation, reserve=reserve)
                proposals = kept_terminal + other

            selected = SurvivalShield().select(observation, proposals, legal)
            action = selected.action if selected else decision.action
            option = selected.option if selected else decision.strategic_option
            bypass = {
                "terminal_win_proposed": bool(terminal_hits),
                "terminal_win_selected": bool(selected and selected.option == "IMMEDIATE_TERMINAL_WIN"),
                "combat_margin_bypass": bool(
                    selected and selected.option == "IMMEDIATE_TERMINAL_WIN" and observation.turn >= 800
                ),
                "risk_gate_bypass": bool(selected and selected.option == "IMMEDIATE_TERMINAL_WIN"),
                "shield_bypass": bool(selected and selected.option == "IMMEDIATE_TERMINAL_WIN"),
            }
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
                    **bypass,
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
            task_b = state.data.get("scout_task_b") or ScoutTask()
            diag["scout_b_source"] = task_b.source
            diag["scout_b_target"] = task_b.target
            diag["scout_b_region_id"] = task_b.region_id
            diag["scout_b_stall"] = task_b.stall
            diag["scout_b_abort_reason"] = task_b.abort_reason
            est = state.data.get("exploration_state") or ExplorationState()
            diag.update(est.to_diagnostics())
        if self.flags.use_hunt_plan:
            diag.update((state.data.get("hunt_plan") or GeneralHuntPlan()).to_diagnostics())
        if terminal_hits:
            diag["terminal_win_candidates"] = len(terminal_hits)
        state.data["diagnostics"] = diag
        decision.new_state = state
        return decision


def create_ablation(name: str) -> HeuristicV2AblationPolicy:
    key = name.strip().lower().replace("-", "_")
    if key not in FLAGS:
        raise KeyError(f"unknown ablation: {name}")
    return HeuristicV2AblationPolicy(FLAGS[key])
