"""Heuristic v1 — map memory, phases, scoreboard and richer proposals."""

from __future__ import annotations

from generals_bot.action import PASS_ACTION, Action
from generals_bot.legal import enumerate_legal_actions
from generals_bot.map_memory import MapMemory
from generals_bot.observation import GameContext, Observation
from generals_bot.policies.base import ActionDecision, PolicyState, Proposal, TraceLevel
from generals_bot.policies.heuristic_config import V1, HeuristicConfig
from generals_bot.protocol import (
    DIRECTIONS,
    OWNER_ME,
    OWNER_NEUTRAL,
    OWNER_OPP,
    TYPE_FOG,
    TYPE_GENERAL,
    TYPE_PLAIN,
)
from generals_bot.risk.shield import SurvivalShield
from generals_bot.rules import DEATHTOUCH_TURN
from generals_bot.scoreboard_inference import scoreboard_view
from generals_bot.turn_phase import TurnPhase, turn_phase, turns_to_draw


class HeuristicV1Policy:
    def __init__(self, config: HeuristicConfig | None = None) -> None:
        self.config = config or V1
        self.policy_id = self.config.name

    def initial_state(self, context: GameContext) -> PolicyState:
        return PolicyState(
            data={
                "player_id": context.player_id,
                "memory": MapMemory.create(context.height, context.width),
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
        memory: MapMemory = state.data["memory"]
        memory.update(observation)
        legal = enumerate_legal_actions(observation)
        proposals = self._propose(observation, memory, legal)
        selected = SurvivalShield().select(observation, proposals, legal)
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
            top_candidates=[p.action for p in proposals[:8]],
            proposals=proposals if trace != TraceLevel.NONE else [],
            shield_result={"selected_module": selected.module if selected else "pass"},
        )

    def _propose(
        self,
        obs: Observation,
        memory: MapMemory,
        legal: list[Action],
    ) -> list[Proposal]:
        cfg = self.config
        phase = turn_phase(obs.turn)
        board = scoreboard_view(obs)
        gen_mask = memory.possible_enemy_general_mask(obs)
        proposals: list[Proposal] = []
        gen = self._find_own_general(obs)

        if gen is not None and self._adjacent_enemy_threat(obs, *gen):
            gr, gc = gen
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
                            score=(1200.0 + obs.army_grid[action.row][action.col])
                            * cfg.defend_weight,
                            confidence=0.95,
                            explanation_code="reinforce_threatened_general",
                        )
                    )

        draw_urgency = 1.0 + max(0, 50 - turns_to_draw(obs.turn)) * 0.05
        dt_active = obs.turn >= DEATHTOUCH_TURN or cfg.prefer_deathtouch

        for action in legal:
            if action.kind != 0:
                continue
            split_pen = 0.15 if action.split == 1 else 0.0
            dr, dc = DIRECTIONS[action.direction]
            nr, nc = action.row + dr, action.col + dc
            src_army = obs.army_grid[action.row][action.col]
            sendable = src_army - 1 if action.split == 0 else src_army // 2
            dest_owner = obs.owner_grid[nr][nc]
            dest_type = obs.type_grid[nr][nc]
            dest_army = obs.army_grid[nr][nc]
            info_gain = float(memory.info_age[nr][nc]) if dest_type == TYPE_FOG else 0.0

            if dest_owner == OWNER_OPP and sendable > dest_army:
                is_general = dest_type == TYPE_GENERAL
                score = (600.0 + sendable - dest_army) * cfg.attack_weight * cfg.aggression
                if is_general:
                    score += 8000.0 * (cfg.deathtouch_weight if dt_active else 1.0)
                if phase == TurnPhase.ENDGAME:
                    score *= draw_urgency
                proposals.append(
                    Proposal(
                        action=action,
                        option="DEATHTOUCH"
                        if is_general and dt_active
                        else ("GENERAL_HUNT" if is_general else "PRESSURE"),
                        module="attack",
                        hard_priority=90 if is_general else 45,
                        score=score - split_pen,
                        confidence=0.85,
                        explanation_code="capture_opponent",
                        explanation_values={
                            "sendable": float(sendable),
                            "land_diff": float(board.land_diff),
                        },
                    )
                )
            elif dest_owner == OWNER_NEUTRAL and dest_type == TYPE_PLAIN and sendable >= 1:
                score = (220.0 + sendable) * cfg.expand_weight
                if phase == TurnPhase.OPENING:
                    score *= 1.25
                proposals.append(
                    Proposal(
                        action=action,
                        option="EXPAND",
                        module="expansion",
                        hard_priority=22,
                        score=score - split_pen,
                        confidence=0.75,
                        explanation_code="expand_plain",
                    )
                )
            elif dest_type == TYPE_FOG and sendable >= 1:
                hunt_bonus = 80.0 if gen_mask[nr][nc] else 0.0
                score = (120.0 + sendable + info_gain + hunt_bonus) * cfg.scout_weight
                proposals.append(
                    Proposal(
                        action=action,
                        option="GENERAL_HUNT" if hunt_bonus else "SCOUT",
                        module="exploration",
                        hard_priority=18,
                        score=score - split_pen,
                        confidence=0.55,
                        explanation_code="information_gain",
                        explanation_values={"info_age": info_gain},
                    )
                )
            elif dest_owner == OWNER_ME and sendable >= 1:
                toward_front = 1.0 if self._is_frontier(obs, nr, nc) else 0.0
                score = (
                    40.0 + float(obs.army_grid[nr][nc]) + 30.0 * toward_front
                ) * cfg.collect_weight
                proposals.append(
                    Proposal(
                        action=action,
                        option="COLLECT",
                        module="collection",
                        hard_priority=12,
                        score=score - split_pen,
                        confidence=0.45,
                        explanation_code="concentrate_mobile_army",
                    )
                )

        for action in legal:
            if action.kind != 2:
                continue
            score = 160.0 * cfg.castle_weight
            if cfg.prefer_castles:
                score += 80.0
            if board.army_diff > 20:
                score += 40.0
            proposals.append(
                Proposal(
                    action=action,
                    option="BUILD",
                    module="castle",
                    hard_priority=28 if cfg.prefer_castles else 24,
                    score=score,
                    confidence=0.6,
                    explanation_code="castle_economics",
                )
            )

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
            if 0 <= nr < obs.height and 0 <= nc < obs.width and obs.owner_grid[nr][nc] == OWNER_OPP:
                return True
        return False

    def _is_frontier(self, obs: Observation, r: int, c: int) -> bool:
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < obs.height and 0 <= nc < obs.width):
                continue
            if obs.owner_grid[nr][nc] != OWNER_ME:
                return True
        return False
