"""Heuristic v0 — legal expansion, collection, basic defence and castle economics."""

from __future__ import annotations

from generals_bot.action import PASS_ACTION, Action
from generals_bot.legal import enumerate_legal_actions
from generals_bot.observation import GameContext, Observation
from generals_bot.policies.base import ActionDecision, PolicyState, Proposal, TraceLevel
from generals_bot.protocol import (
    DIRECTIONS,
    OWNER_ME,
    OWNER_NEUTRAL,
    OWNER_OPP,
    TYPE_FOG,
    TYPE_GENERAL,
    TYPE_PLAIN,
    TYPE_STRUCTURE_IN_FOG,
)
from generals_bot.risk.shield import SurvivalShield


class HeuristicV0Policy:
    """Deterministic heuristic v0."""

    policy_id = "heuristic_v0"

    def initial_state(self, context: GameContext) -> PolicyState:
        return PolicyState(
            data={
                "player_id": context.player_id,
                "height": context.height,
                "width": context.width,
            }
        )

    def act(
        self,
        observation: Observation,
        state: PolicyState,
        *,
        deterministic: bool,
        trace: TraceLevel,
        deadline: float | None,
    ) -> ActionDecision:
        legal = enumerate_legal_actions(observation)
        proposals = self._propose(observation, legal)
        shield = SurvivalShield()
        selected = shield.select(observation, proposals, legal)
        option = selected.option if selected else "WAIT"
        action = selected.action if selected else PASS_ACTION
        return ActionDecision(
            action=action,
            new_state=state,
            strategic_option=option,
            option_distribution={option: 1.0},
            policy_id=self.policy_id,
            confidence=selected.confidence if selected else 1.0,
            legal_action_count=len(legal),
            top_candidates=[p.action for p in proposals[:5]],
            proposals=proposals if trace != TraceLevel.NONE else [],
            shield_result={"selected_module": selected.module if selected else "pass"},
        )

    def _propose(self, obs: Observation, legal: list[Action]) -> list[Proposal]:
        proposals: list[Proposal] = []

        # General defence: reinforce general if threatened by adjacent enemy.
        gen = self._find_own_general(obs)
        if gen is not None:
            gr, gc = gen
            if self._adjacent_enemy_threat(obs, gr, gc):
                for action in legal:
                    if action.kind != 0:
                        continue
                    dr, dc = DIRECTIONS[action.direction]
                    if (action.row + dr, action.col + dc) == (gr, gc):
                        proposals.append(
                            Proposal(
                                action=action,
                                option="DEFEND",
                                module="defence",
                                hard_priority=100,
                                score=1000.0 + obs.army_grid[action.row][action.col],
                                confidence=0.9,
                                explanation_code="reinforce_threatened_general",
                            )
                        )

        # Capture opponent / expand into visible neutral / scout fog
        for action in legal:
            if action.kind != 0 or action.split != 0:
                continue
            dr, dc = DIRECTIONS[action.direction]
            nr, nc = action.row + dr, action.col + dc
            src_army = obs.army_grid[action.row][action.col]
            dest_owner = obs.owner_grid[nr][nc]
            dest_type = obs.type_grid[nr][nc]
            dest_army = obs.army_grid[nr][nc]
            sendable = src_army - 1

            if dest_owner == OWNER_OPP and sendable > dest_army:
                score = 500.0 + sendable - dest_army
                if dest_type == TYPE_GENERAL:
                    score += 5000.0
                proposals.append(
                    Proposal(
                        action=action,
                        option="ATTACK" if dest_type == TYPE_GENERAL else "PRESSURE",
                        module="attack" if dest_type == TYPE_GENERAL else "pressure",
                        hard_priority=80 if dest_type == TYPE_GENERAL else 40,
                        score=score,
                        confidence=0.8,
                        explanation_code="capture_opponent",
                        explanation_values={
                            "sendable": float(sendable),
                            "dest_army": float(dest_army),
                        },
                    )
                )
            elif dest_owner == OWNER_NEUTRAL and dest_type == TYPE_PLAIN and sendable >= 1:
                proposals.append(
                    Proposal(
                        action=action,
                        option="EXPAND",
                        module="expansion",
                        hard_priority=20,
                        score=200.0 + sendable,
                        confidence=0.7,
                        explanation_code="expand_plain",
                    )
                )
            elif dest_type == TYPE_FOG and sendable >= 1:
                proposals.append(
                    Proposal(
                        action=action,
                        option="SCOUT",
                        module="exploration",
                        hard_priority=15,
                        score=100.0 + sendable,
                        confidence=0.5,
                        explanation_code="scout_fog",
                    )
                )
            elif dest_owner == OWNER_ME and sendable >= 1:
                # Collection toward larger stacks / general
                proposals.append(
                    Proposal(
                        action=action,
                        option="COLLECT",
                        module="collection",
                        hard_priority=10,
                        score=50.0 + float(obs.army_grid[nr][nc]),
                        confidence=0.4,
                        explanation_code="friendly_merge",
                    )
                )

        # Basic castle economics: build when affordable and far enough
        for action in legal:
            if action.kind != 2:
                continue
            proposals.append(
                Proposal(
                    action=action,
                    option="BUILD",
                    module="castle",
                    hard_priority=25,
                    score=150.0,
                    confidence=0.55,
                    explanation_code="affordable_castle",
                )
            )

        # Always allow WAIT/pass as lowest priority
        proposals.append(
            Proposal(
                action=PASS_ACTION,
                option="WAIT",
                module="pass",
                hard_priority=0,
                score=0.0,
                confidence=1.0,
                explanation_code="safe_pass",
            )
        )
        return proposals

    def _find_own_general(self, obs: Observation) -> tuple[int, int] | None:
        for r in range(obs.height):
            for c in range(obs.width):
                if obs.owner_grid[r][c] == OWNER_ME and obs.type_grid[r][c] == TYPE_GENERAL:
                    return r, c
        return None

    def _adjacent_enemy_threat(self, obs: Observation, r: int, c: int) -> bool:
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < obs.height and 0 <= nc < obs.width:
                if obs.owner_grid[nr][nc] == OWNER_OPP:
                    return True
                if obs.type_grid[nr][nc] == TYPE_STRUCTURE_IN_FOG:
                    return True
        return False
