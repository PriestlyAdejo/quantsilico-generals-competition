"""Ablation wrappers around frozen v2f reference with optional modules."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from generals_bot.action import Action
from generals_bot.legal import enumerate_legal_actions
from generals_bot.map_memory import MapMemory
from generals_bot.observation import GameContext, Observation
from generals_bot.policies.attack_commitment import (
    DEFAULT_ATTACK_READINESS,
    AttackCommitmentState,
    AttackReadinessConfig,
    evaluate_readiness_ok,
    filter_proposals_for_commitment,
    mobile_army_total,
    update_attack_commitment,
)
from generals_bot.policies.base import ActionDecision, PolicyState, Proposal, TraceLevel
from generals_bot.policies.bounded_scout import BoundedScoutAssigner, ScoutTask
from generals_bot.policies.exploration_planner import ExplorationState
from generals_bot.policies.general_garrison import (
    filter_general_stripping,
    find_own_general,
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
from generals_bot.policies.phase_controller_v2f import StrategicPhase
from generals_bot.policies.terminal_win_oracle import immediate_terminal_win_proposals
from generals_bot.policies.threat_assessment import ThreatMemory, assess_threat
from generals_bot.protocol import OWNER_OPP
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
    use_attack_commitment: bool = False

    def config_hash(self) -> str:
        payload_obj = asdict(self)
        # Preserve historical hashes for candidates that do not use newer flags.
        for optional in (
            "use_intercept_v2",
            "use_terminal_oracle",
            "use_hunt_plan",
            "use_persistent_explore",
            "use_dual_scout",
            "use_attack_commitment",
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
V2F_TACTICAL_ATTACK_V2 = "heuristic_v2f_tactical_attack_v2"


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
    V2F_TACTICAL_ATTACK_V2: AblationFlags(
        name=V2F_TACTICAL_ATTACK_V2,
        use_planner=True,
        use_terminal_oracle=True,
        use_hunt_plan=True,
        use_credible_threat=True,
        use_persistent_explore=True,
        use_dual_scout=True,
        use_attack_commitment=True,
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
        if self.flags.use_attack_commitment:
            state.data["attack_commitment"] = AttackCommitmentState.NONE.value
            state.data["attack_prepare_dwell"] = 0
            state.data["attack_retreat_dwell"] = 0
            state.data["eg_belief_turn"] = -1
            state.data["eg_confidence"] = 0.0
            state.data["attack_readiness_config"] = DEFAULT_ATTACK_READINESS
        return state

    def _update_attack_commitment_state(
        self,
        observation: Observation,
        state: PolicyState,
        *,
        known_eg: tuple[int, int] | None,
        emergency: bool,
        terminal_hits: list[Proposal],
    ) -> AttackCommitmentState:
        cfg: AttackReadinessConfig = state.data.get("attack_readiness_config") or DEFAULT_ATTACK_READINESS
        prev_raw = state.data.get("attack_commitment") or AttackCommitmentState.NONE.value
        try:
            prev = AttackCommitmentState(prev_raw)
        except ValueError:
            prev = AttackCommitmentState.NONE

        # Belief tracking: refresh when EG cell is visible enemy-owned.
        eg_confidence = float(state.data.get("eg_confidence") or 0.0)
        eg_belief_turn = int(state.data.get("eg_belief_turn") or -1)
        if known_eg is not None:
            er, ec = known_eg
            visible = (
                0 <= er < observation.height
                and 0 <= ec < observation.width
                and observation.owner_grid[er][ec] == OWNER_OPP
            )
            if visible:
                eg_belief_turn = observation.turn
                eg_confidence = 1.0
            elif eg_belief_turn < 0:
                eg_belief_turn = observation.turn
                eg_confidence = max(eg_confidence, 0.7)
            else:
                eg_confidence = max(0.0, eg_confidence - 0.01)
        else:
            eg_confidence = 0.0
            eg_belief_turn = -1
        state.data["eg_confidence"] = eg_confidence
        state.data["eg_belief_turn"] = eg_belief_turn
        belief_age = (
            0 if eg_belief_turn < 0 else max(0, observation.turn - eg_belief_turn)
        )

        plan: GeneralHuntPlan = state.data.get("hunt_plan") or GeneralHuntPlan()
        gathering = str(plan.blocked_reason or "").startswith("gather_until_")
        route_illegal = plan.blocked_reason in {
            "no_reachable_route",
            "next_edge_invalid",
            "no_attack_source",
        } or (known_eg is not None and not plan.route and plan.active)
        route_length = len(plan.route)

        eg_captured = False
        if known_eg is not None:
            er, ec = known_eg
            own = find_own_general(observation)
            if own is not None and (er, ec) == own:
                eg_captured = True
            elif (
                0 <= er < observation.height
                and 0 <= ec < observation.width
                and observation.owner_grid[er][ec] != OWNER_OPP
                and belief_age > cfg.belief_age_max
            ):
                eg_captured = False  # stale handled via belief_age

        mobile = mobile_army_total(observation)
        prepare_dwell = int(state.data.get("attack_prepare_dwell") or 0)
        retreat_dwell = int(state.data.get("attack_retreat_dwell") or 0)
        if prev == AttackCommitmentState.PREPARE:
            prepare_dwell += 1
        else:
            prepare_dwell = 0 if prev != AttackCommitmentState.COMMIT else prepare_dwell
        if prev == AttackCommitmentState.RETREAT:
            retreat_dwell += 1
        else:
            retreat_dwell = 0

        readiness_ok = evaluate_readiness_ok(
            cfg=cfg,
            eg_confidence=eg_confidence,
            belief_age=belief_age,
            mobile_army=mobile,
            route_length=route_length,
            gathering=gathering,
            own_general_threatened=emergency,
            prepare_dwell=prepare_dwell if prev == AttackCommitmentState.PREPARE else cfg.dwell_turns_prepare_to_commit,
            counterattack_risk=0.0,
        )
        # Hysteresis: leave RETREAT only after dwell.
        if prev == AttackCommitmentState.RETREAT and retreat_dwell < cfg.dwell_turns_retreat:
            readiness_ok = False

        # Combat margin: material deficit vs visible EG army along hunt source.
        combat_margin_negative = False
        if prev == AttackCommitmentState.COMMIT and plan.source is not None and known_eg is not None:
            sr, sc = plan.source
            src_army = int(observation.army_grid[sr][sc]) if observation.owner_grid[sr][sc] else 0
            er, ec = known_eg
            eg_army = (
                int(observation.army_grid[er][ec])
                if observation.owner_grid[er][ec] == OWNER_OPP
                else 0
            )
            if src_army - 1 + cfg.combat_margin < eg_army:
                combat_margin_negative = True

        convert_ready = bool(
            prev == AttackCommitmentState.COMMIT
            and known_eg is not None
            and not gathering
            and mobile >= cfg.min_attack_stack * 1.5
        )

        nxt = update_attack_commitment(
            prev,
            known_eg=known_eg,
            eg_confidence=eg_confidence,
            belief_age=belief_age,
            readiness_ok=readiness_ok,
            emergency=emergency,
            route_illegal=route_illegal,
            eg_captured=eg_captured,
            terminal=bool(terminal_hits),
            combat_margin_negative=combat_margin_negative,
            convert_ready=convert_ready,
            turn=observation.turn,
        )
        state.data["attack_commitment"] = nxt.value
        state.data["attack_prepare_dwell"] = prepare_dwell
        state.data["attack_retreat_dwell"] = retreat_dwell
        state.data["attack_readiness_ok"] = readiness_ok
        return nxt

    def generate_proposals(
        self,
        observation: Observation,
        state: PolicyState,
        *,
        deadline: float | None = None,
    ) -> tuple[list[Proposal], PolicyState, list[Action]]:
        """Assemble ranked proposals without SurvivalShield selection.

        Shared by ``act`` and hybrid BC rankers — do not duplicate this path.
        """
        memory: MapMemory = state.data["memory"]
        prev_seen = sum(1 for row in memory.ever_seen for v in row if v)

        threat_mem: ThreatMemory = state.data.get("threat") or ThreatMemory()
        threat, threat_mem = assess_threat(observation, threat_mem)
        state.data["threat"] = threat_mem

        decision = self._base.act(
            observation, state, deterministic=True, trace=TraceLevel.DECISION, deadline=deadline
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
            # With attack commitment: keep BUILD for PREPARE contextual castles; COMMIT strips later.
            if not emergency_for_hunt:
                filtered = []
                for p in proposals:
                    if p.option == "IMMEDIATE_TERMINAL_WIN":
                        filtered.append(p)
                        continue
                    # CASTLE is a legacy alias; generators emit BUILD.
                    strip_opts = {"EXPAND", "CASTLE", "BUILD", "SCOUT"}
                    if self.flags.use_attack_commitment:
                        strip_opts = {"EXPAND", "SCOUT"}
                    if p.option in strip_opts and p.hard_priority < 100:
                        continue
                    if p.option == "DEFEND" and p.hard_priority <= 100:
                        continue
                    filtered.append(p)
                proposals = filtered

        terminal_hits: list[Proposal] = []
        if self.flags.use_terminal_oracle:
            terminal_hits = immediate_terminal_win_proposals(
                observation, known_eg, legal=legal, hard_priority=110
            )
            proposals.extend(terminal_hits)

        if self.flags.use_attack_commitment:
            commitment = self._update_attack_commitment_state(
                observation,
                state,
                known_eg=known_eg,
                emergency=emergency_for_hunt,
                terminal_hits=terminal_hits,
            )
            proposals = filter_proposals_for_commitment(
                proposals,
                commitment,
                known_eg=known_eg,
                emergency=emergency_for_hunt,
            )
            # Soft-gate: do not extend StrategicPhase. If phase is GENERAL_HUNT while
            # PREPARE, rewrite diagnostics phase_reason and demote approaches (above).
            phase_val = state.data.get("phase")
            if (
                commitment == AttackCommitmentState.PREPARE
                and phase_val == StrategicPhase.GENERAL_HUNT.value
            ):
                state.data["phase_reason"] = "attack_prepare_soft_gate"
                diag = dict(state.data.get("diagnostics") or {})
                diag["phase_reason"] = "attack_prepare_soft_gate"
                diag["attack_commitment"] = commitment.value
                state.data["diagnostics"] = diag

        if proposals and self.flags.use_garrison:
            reserve = garrison_reserve(turn=observation.turn, threatened=threat.emergency)
            kept_terminal = [p for p in proposals if p.option == "IMMEDIATE_TERMINAL_WIN"]
            other = [p for p in proposals if p.option != "IMMEDIATE_TERMINAL_WIN"]
            other = filter_general_stripping(other, observation, reserve=reserve)
            proposals = kept_terminal + other

        state.data["_ablation_base"] = {
            "action": decision.action,
            "strategic_option": decision.strategic_option,
            "confidence": decision.confidence,
            "shield_result": decision.shield_result,
            "terminal_hits": len(terminal_hits),
            "threat_emergency": threat.emergency,
            "threat_caution": threat.caution,
        }
        return proposals, state, legal

    def act(
        self,
        observation,
        state,
        *,
        deterministic: bool,
        trace: TraceLevel,
        deadline: float | None,
    ) -> ActionDecision:
        del deterministic  # proposal path is deterministic; shield selects.
        proposals, state, legal = self.generate_proposals(observation, state, deadline=deadline)
        base = state.data.get("_ablation_base") or {}
        terminal_hits_n = int(base.get("terminal_hits") or 0)
        threat_emergency = bool(base.get("threat_emergency"))
        threat_caution = bool(base.get("threat_caution"))

        if proposals:
            selected = SurvivalShield().select(observation, proposals, legal)
            action = selected.action if selected else base.get("action")
            option = selected.option if selected else base.get("strategic_option", "WAIT")
            bypass = {
                "terminal_win_proposed": terminal_hits_n > 0,
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
                confidence=selected.confidence if selected else float(base.get("confidence") or 1.0),
                legal_action_count=len(legal),
                top_candidates=[p.action for p in proposals[:8]],
                proposals=proposals if trace != TraceLevel.NONE else [],
                shield_result={
                    **(base.get("shield_result") or {}),
                    "ablation": self.flags.name,
                    "config_hash": self.config_hash,
                    "selected_module": selected.module if selected else "base",
                    **bypass,
                },
            )
        else:
            decision = ActionDecision(
                action=base.get("action"),
                new_state=state,
                strategic_option=str(base.get("strategic_option") or "WAIT"),
                policy_id=self.policy_id,
                confidence=float(base.get("confidence") or 1.0),
                legal_action_count=len(legal),
                proposals=[],
                shield_result={
                    **(base.get("shield_result") or {}),
                    "ablation": self.flags.name,
                    "config_hash": self.config_hash,
                },
            )

        diag = dict(state.data.get("diagnostics") or {})
        diag["ablation"] = self.flags.name
        diag["config_hash"] = self.config_hash
        diag["emergency_active"] = threat_emergency
        diag["caution_active"] = threat_caution
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
        if self.flags.use_attack_commitment:
            diag["attack_commitment"] = state.data.get("attack_commitment")
            diag["attack_readiness_ok"] = state.data.get("attack_readiness_ok")
            diag["attack_prepare_dwell"] = state.data.get("attack_prepare_dwell")
            diag["attack_retreat_dwell"] = state.data.get("attack_retreat_dwell")
        if terminal_hits_n:
            diag["terminal_win_candidates"] = terminal_hits_n
        state.data["diagnostics"] = diag
        decision.new_state = state
        return decision


def create_ablation(name: str) -> HeuristicV2AblationPolicy:
    key = name.strip().lower().replace("-", "_")
    if key not in FLAGS:
        raise KeyError(f"unknown ablation: {name}")
    return HeuristicV2AblationPolicy(FLAGS[key])
