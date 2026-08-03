"""Heuristic v2 qualifier — conversion, Deathtouch and defence-first vs Expander."""

from __future__ import annotations

from generals_bot.action import KIND_BUILD, KIND_MOVE, PASS_ACTION, Action
from generals_bot.castle_cost import castle_price_at, own_structures
from generals_bot.legal import enumerate_legal_actions
from generals_bot.map_memory import MapMemory
from generals_bot.observation import GameContext, Observation
from generals_bot.policies.base import ActionDecision, PolicyState, Proposal, TraceLevel
from generals_bot.policies.heuristic_config import HeuristicConfig
from generals_bot.policies.phase_controller import (
    StrategicPhase,
    is_dominant_position,
    select_phase,
)
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
from generals_bot.rules import DEATHTOUCH_TURN, DRAW_TURN

V2 = HeuristicConfig(
    name="heuristic_v2_qualifier",
    expand_weight=1.0,
    attack_weight=1.3,
    defend_weight=1.4,
    scout_weight=1.2,
    collect_weight=1.15,
    castle_weight=0.85,
    deathtouch_weight=3.0,
    aggression=1.15,
    prefer_castles=False,
    prefer_deathtouch=True,
)


class HeuristicV2QualifierPolicy:
    """Deterministic qualification-oriented heuristic.

    Primary fixes vs v1:
    - Deathtouch: legal touch onto enemy general after turn 800 without army margin
    - Explicit phase controller with mandatory late-game transitions
    - Dominant-position conversion (stop farming neutrals)
    - Stronger concentration and emergency defence
    - Selective early castle economics (not forced)
    """

    def __init__(self, config: HeuristicConfig | None = None) -> None:
        self.config = config or V2
        self.policy_id = self.config.name

    def initial_state(self, context: GameContext) -> PolicyState:
        return PolicyState(
            data={
                "player_id": context.player_id,
                "memory": MapMemory.create(context.height, context.width),
                "phase": StrategicPhase.OPENING.value,
                "phase_reason": "init",
                "enemy_contact": False,
                "known_enemy_general": None,
                "castles_built": 0,
                "last_newly_scouted_turn": 0,
                "diagnostics": {},
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
        prev_seen = sum(1 for row in memory.ever_seen for v in row if v)
        memory.update(observation)
        seen_now = sum(1 for row in memory.ever_seen for v in row if v)
        if seen_now > prev_seen:
            state.data["last_newly_scouted_turn"] = observation.turn

        # Track known enemy general
        eg = self._find_enemy_general(observation)
        if eg is not None:
            state.data["known_enemy_general"] = eg
        known_eg = state.data.get("known_enemy_general")

        if self._has_enemy_visible(observation):
            state.data["enemy_contact"] = True
            # Remember centroid of visible enemy for fog focus when they slip into fog
            cells = [
                (r, c)
                for r in range(observation.height)
                for c in range(observation.width)
                if observation.owner_grid[r][c] == OWNER_OPP
            ]
            if cells:
                state.data["last_enemy_sighting"] = (
                    sum(r for r, _ in cells) // len(cells),
                    sum(c for _, c in cells) // len(cells),
                )

        gen = self._find_own_general(observation)
        threatened = bool(gen and self._general_threat_level(observation, *gen) > 0)
        gen_mask = memory.possible_enemy_general_mask(observation)
        mask_size = sum(1 for row in gen_mask for v in row if v)
        mobile, frontier_mobile, stranded, concentration = self._army_stats(observation)
        dominant = is_dominant_position(
            my_land=observation.my_land, opp_land=observation.opp_land
        )
        phase, reason = select_phase(
            observation,
            prev=StrategicPhase(state.data.get("phase", StrategicPhase.OPENING.value)),
            enemy_contact=bool(state.data["enemy_contact"]),
            enemy_general_known=known_eg is not None,
            own_general_threatened=threatened,
            dominant=dominant,
            mobile_ratio=concentration,
            candidate_mask_size=mask_size,
        )
        state.data["phase"] = phase.value
        state.data["phase_reason"] = reason

        legal = enumerate_legal_actions(observation)
        proposals = self._propose(
            observation,
            memory,
            legal,
            phase=phase,
            known_eg=known_eg,
            gen_mask=gen_mask,
            castles_built=int(state.data.get("castles_built", 0)),
            last_enemy_sighting=state.data.get("last_enemy_sighting"),
        )
        selected = SurvivalShield().select(observation, proposals, legal)
        option = selected.option if selected else "WAIT"
        action = selected.action if selected else PASS_ACTION
        if action.kind == KIND_BUILD:
            state.data["castles_built"] = int(state.data.get("castles_built", 0)) + 1

        state.data["diagnostics"] = {
            "phase": phase.value,
            "phase_reason": reason,
            "mobile_army": mobile,
            "frontier_mobile_army": frontier_mobile,
            "largest_stranded_stack": stranded,
            "army_concentration_ratio": concentration,
            "candidate_mask_size": mask_size,
            "dominant": dominant,
            "known_enemy_general": known_eg,
            "castle_proposals": sum(1 for p in proposals if p.module == "castle"),
        }

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
            shield_result={
                "selected_module": selected.module if selected else "pass",
                "phase": phase.value,
                "phase_reason": reason,
            },
        )

    def _propose(
        self,
        obs: Observation,
        memory: MapMemory,
        legal: list[Action],
        *,
        phase: StrategicPhase,
        known_eg: tuple[int, int] | None,
        gen_mask: list[list[bool]],
        castles_built: int,
        last_enemy_sighting: tuple[int, int] | None = None,
    ) -> list[Proposal]:
        cfg = self.config
        proposals: list[Proposal] = []
        gen = self._find_own_general(obs)
        fog_targets = memory.enemy_frontier_fog_targets(obs)
        # If scoreboard says opponent still has land but fog hides them, aim at last sighting / best fog
        hunt_anchor = known_eg or last_enemy_sighting
        if hunt_anchor is None and fog_targets:
            hunt_anchor = (fog_targets[0][0], fog_targets[0][1])
        dt = obs.turn >= DEATHTOUCH_TURN
        late = obs.turn >= 1050
        very_late = obs.turn >= 1150
        force_hunt = phase in {
            StrategicPhase.CONVERSION,
            StrategicPhase.GENERAL_HUNT,
            StrategicPhase.DEATHTOUCH_HUNT,
            StrategicPhase.DRAW_AVOIDANCE,
        }
        # Late unresolved games: hunt even without conversion phase label
        if (dt or late) and obs.opp_land > 0:
            force_hunt = True
        expand_scale = 0.15 if force_hunt else (0.05 if very_late else 1.0)
        collect_scale = 1.8 if phase == StrategicPhase.CONSOLIDATION else 1.2
        hunt_scale = 3.0 if force_hunt else 1.0

        # Focused fog-sweep after contact / late: aim stacks at ranked fog near enemy.
        # Disabled once the enemy general cell is known — do not divert from the kill route.
        if (
            force_hunt
            and known_eg is None
            and fog_targets
            and phase not in {
                StrategicPhase.OPENING,
                StrategicPhase.EXPANSION,
            }
        ):
            ranked = fog_targets
            if last_enemy_sighting is not None:
                lr, lc = last_enemy_sighting
                ranked = sorted(
                    fog_targets,
                    key=lambda t: (-t[2], abs(t[0] - lr) + abs(t[1] - lc)),
                )
            # Late: rotate among top-3 to escape local fog loops
            if late or very_late:
                pick = ranked[obs.turn % min(3, len(ranked))]
            else:
                pick = ranked[0]
            hunt_anchor = (pick[0], pick[1])
            ar, ac = hunt_anchor
            near_fog = {
                (t[0], t[1])
                for t in ranked[:8]
                if abs(t[0] - ar) + abs(t[1] - ac) <= 6
            }
            for action in legal:
                if action.kind != KIND_MOVE or action.split == 1:
                    continue
                dr, dc = DIRECTIONS[action.direction]
                nr, nc = action.row + dr, action.col + dc
                src_army = obs.army_grid[action.row][action.col]
                sendable = src_army - 1
                if sendable < 1:
                    continue
                dist_src = abs(action.row - ar) + abs(action.col - ac)
                dist_dst = abs(nr - ar) + abs(nc - ac)
                into_enemy = obs.owner_grid[nr][nc] == OWNER_OPP
                into_focus_fog = obs.type_grid[nr][nc] == TYPE_FOG and (
                    (nr, nc) in near_fog
                    or (nr, nc) == (ar, ac)
                    or abs(nr - ar) + abs(nc - ac) <= 2
                )
                if into_enemy or into_focus_fog:
                    proposals.append(
                        Proposal(
                            action=action,
                            option="GENERAL_HUNT",
                            module="fog_sweep",
                            hard_priority=90,
                            score=(1200.0 + sendable * 10 + 50.0 * max(0, dist_src - dist_dst))
                            * hunt_scale,
                            confidence=0.85,
                            explanation_code="sweep_into_enemy_or_fog",
                        )
                    )
                elif dist_dst < dist_src and sendable >= 5:
                    proposals.append(
                        Proposal(
                            action=action,
                            option="GENERAL_HUNT",
                            module="fog_sweep",
                            hard_priority=60 if late else 45,
                            score=(400.0 + sendable * 5 + 30.0 * (dist_src - dist_dst))
                            * hunt_scale,
                            confidence=0.7,
                            explanation_code="approach_enemy_region",
                        )
                    )

        # Emergency / proactive defence
        if gen is not None:
            threat = self._general_threat_level(obs, *gen)
            if threat > 0 or phase == StrategicPhase.EMERGENCY_DEFENCE:
                gr, gc = gen
                for action in legal:
                    if action.kind != KIND_MOVE:
                        continue
                    dr, dc = DIRECTIONS[action.direction]
                    nr, nc = action.row + dr, action.col + dc
                    # Never leave the general with a stripping move during threat
                    if (action.row, action.col) == (gr, gc):
                        continue
                    dist_src = abs(action.row - gr) + abs(action.col - gc)
                    dist_dst = abs(nr - gr) + abs(nc - gc)
                    if (nr, nc) == (gr, gc) or dist_dst < dist_src:
                        proposals.append(
                            Proposal(
                                action=action,
                                option="DEFEND",
                                module="defence",
                                hard_priority=100,
                                score=(2000.0 + obs.army_grid[action.row][action.col] * 10)
                                * cfg.defend_weight
                                / max(1, dist_dst + 1),
                                confidence=0.95,
                                explanation_code="reinforce_threatened_general",
                                explanation_values={"threat": float(threat)},
                            )
                        )

        # Known-general Deathtouch / hunt routes (including insufficient army after 800)
        if known_eg is not None:
            er, ec = known_eg
            for action in legal:
                if action.kind != KIND_MOVE:
                    continue
                dr, dc = DIRECTIONS[action.direction]
                nr, nc = action.row + dr, action.col + dc
                src_army = obs.army_grid[action.row][action.col]
                sendable = src_army - 1 if action.split == 0 else src_army // 2
                if sendable < 1:
                    continue
                if (nr, nc) == (er, ec):
                    # Direct touch
                    need_margin = not dt
                    dest_army = obs.army_grid[nr][nc] if obs.owner_grid[nr][nc] == OWNER_OPP else 0
                    if need_margin and sendable <= dest_army:
                        continue
                    proposals.append(
                        Proposal(
                            action=action,
                            option="DEATHTOUCH" if dt else "GENERAL_HUNT",
                            module="deathtouch" if dt else "general_hunt",
                            hard_priority=95,
                            score=(20000.0 + sendable) * cfg.deathtouch_weight,
                            confidence=0.99,
                            explanation_code="touch_enemy_general",
                        )
                    )
                else:
                    # Move closer to known general — hard priority above fog sweep
                    dist_src = abs(action.row - er) + abs(action.col - ec)
                    dist_dst = abs(nr - er) + abs(nc - ec)
                    if dist_dst < dist_src:
                        proposals.append(
                            Proposal(
                                action=action,
                                option="DEATHTOUCH" if dt else "GENERAL_HUNT",
                                module="general_hunt",
                                hard_priority=93,
                                score=(900.0 + sendable * 5 + 50.0 * (dist_src - dist_dst))
                                * hunt_scale,
                                confidence=0.8,
                                explanation_code="approach_enemy_general",
                            )
                        )

        for action in legal:
            if action.kind != KIND_MOVE:
                continue
            split_pen = 0.1 if action.split == 1 else 0.0
            dr, dc = DIRECTIONS[action.direction]
            nr, nc = action.row + dr, action.col + dc
            src_army = obs.army_grid[action.row][action.col]
            sendable = src_army - 1 if action.split == 0 else src_army // 2
            if sendable < 1:
                continue
            dest_owner = obs.owner_grid[nr][nc]
            dest_type = obs.type_grid[nr][nc]
            dest_army = obs.army_grid[nr][nc]
            info_gain = float(memory.info_age[nr][nc]) if dest_type == TYPE_FOG else 0.0
            on_candidate = bool(gen_mask[nr][nc])

            # Opponent attacks — Deathtouch exception for generals
            if dest_owner == OWNER_OPP:
                is_general = dest_type == TYPE_GENERAL
                can_hit = sendable > dest_army or (dt and is_general and sendable >= 1)
                if not can_hit:
                    continue
                score = (700.0 + sendable - dest_army) * cfg.attack_weight * cfg.aggression
                if is_general:
                    score += 10000.0 * cfg.deathtouch_weight
                if phase in {
                    StrategicPhase.CONVERSION,
                    StrategicPhase.CONTACT,
                    StrategicPhase.DRAW_AVOIDANCE,
                    StrategicPhase.DEATHTOUCH_HUNT,
                }:
                    score *= 1.8
                    # Compress remaining enemy pocket
                    score += 200.0
                proposals.append(
                    Proposal(
                        action=action,
                        option="DEATHTOUCH"
                        if is_general and dt
                        else ("GENERAL_HUNT" if is_general else "PRESSURE"),
                        module="attack",
                        hard_priority=92 if is_general else (75 if phase == StrategicPhase.CONVERSION else 50),
                        score=score - split_pen,
                        confidence=0.9,
                        explanation_code="capture_opponent",
                    )
                )
            elif dest_type == TYPE_FOG and sendable >= 1:
                hunt_bonus = 200.0 * hunt_scale if on_candidate else 0.0
                frontier_bonus = 0.0
                for er, ec, pri in fog_targets[:12]:
                    if (nr, nc) == (er, ec):
                        frontier_bonus = float(pri) * 3.0
                        break
                if late:
                    hunt_bonus += 150.0
                if very_late:
                    hunt_bonus += 300.0
                far_bonus = 0.0
                if gen is not None and not force_hunt:
                    far_bonus = 8.0 * (abs(nr - gen[0]) + abs(nc - gen[1]))
                score = (
                    140.0 + sendable + info_gain + hunt_bonus + frontier_bonus + far_bonus
                ) * cfg.scout_weight
                hard = 20
                if force_hunt and (on_candidate or frontier_bonus > 0):
                    hard = 88
                elif on_candidate and (dt or late):
                    hard = 70
                elif not force_hunt and phase in {
                    StrategicPhase.OPENING,
                    StrategicPhase.EXPANSION,
                    StrategicPhase.CONSOLIDATION,
                }:
                    hard = 21  # deep scout competes on score, not above expand
                proposals.append(
                    Proposal(
                        action=action,
                        option="GENERAL_HUNT" if on_candidate or force_hunt else "SCOUT",
                        module="exploration",
                        hard_priority=hard,
                        score=score - split_pen,
                        confidence=0.6,
                        explanation_code="scout_or_hunt_fog",
                        explanation_values={"candidate": float(on_candidate), "frontier": frontier_bonus},
                    )
                )
            elif dest_owner == OWNER_NEUTRAL and dest_type == TYPE_PLAIN and sendable >= 1:
                if expand_scale < 0.2 and phase != StrategicPhase.OPENING:
                    score = (80.0 + sendable) * cfg.expand_weight * expand_scale
                else:
                    score = (220.0 + sendable) * cfg.expand_weight * expand_scale
                if gen is not None:
                    if abs(nr - gen[0]) + abs(nc - gen[1]) > abs(action.row - gen[0]) + abs(
                        action.col - gen[1]
                    ):
                        score += 40.0
                proposals.append(
                    Proposal(
                        action=action,
                        option="EXPAND",
                        module="expansion",
                        hard_priority=10 if expand_scale < 0.5 else 22,
                        score=score - split_pen,
                        confidence=0.7,
                        explanation_code="expand_plain",
                    )
                )
            elif dest_owner == OWNER_ME and sendable >= 1:
                toward_front = 1.0 if self._is_frontier(obs, nr, nc) else 0.0
                toward_enemy = 0.0
                if known_eg is not None:
                    er, ec = known_eg
                    if abs(nr - er) + abs(nc - ec) < abs(action.row - er) + abs(action.col - ec):
                        toward_enemy = 1.0
                score = (
                    50.0
                    + float(obs.army_grid[nr][nc])
                    + 40.0 * toward_front
                    + 80.0 * toward_enemy
                ) * cfg.collect_weight * collect_scale
                proposals.append(
                    Proposal(
                        action=action,
                        option="COLLECT",
                        module="collection",
                        hard_priority=35 if phase == StrategicPhase.CONSOLIDATION else 15,
                        score=score - split_pen,
                        confidence=0.55,
                        explanation_code="concentrate_mobile_army",
                    )
                )

        # Selective castles — skip in defence / late conversion
        if (
            phase not in {
                StrategicPhase.EMERGENCY_DEFENCE,
                StrategicPhase.DEATHTOUCH_HUNT,
                StrategicPhase.DRAW_AVOIDANCE,
                StrategicPhase.CONVERSION,
            }
            and castles_built < 2
            and obs.turn < 600
            and not late
        ):
            structures = own_structures(obs)
            for action in legal:
                if action.kind != KIND_BUILD:
                    continue
                price = castle_price_at(action.row, action.col, structures)
                army = obs.army_grid[action.row][action.col]
                remaining = army - price
                turns_left = max(1, DRAW_TURN - obs.turn)
                payback = price * 2  # ~1 army / 2 turns
                if remaining < 3:
                    continue
                if payback > turns_left * 0.5:
                    continue
                if self._is_frontier(obs, action.row, action.col):
                    continue  # prefer interior
                score = (180.0 + remaining - price * 0.2) * cfg.castle_weight
                if castles_built == 0 and obs.turn < 200:
                    score += 40.0
                proposals.append(
                    Proposal(
                        action=action,
                        option="BUILD",
                        module="castle",
                        hard_priority=26,
                        score=score,
                        confidence=0.55,
                        explanation_code="selective_castle",
                        explanation_values={
                            "price": float(price),
                            "payback_turns": float(payback),
                            "remaining": float(remaining),
                        },
                    )
                )

        proposals.append(
            Proposal(
                action=PASS_ACTION,
                option="WAIT",
                module="pass",
                hard_priority=0,
                score=-1.0 if very_late else 0.0,
                confidence=1.0,
                explanation_code="safe_pass",
            )
        )

        # Standing garrison: drop proposals that strip the general below reserve.
        # Opening must still leave the general to establish a route (low reserve).
        if gen is not None:
            gr, gc = gen
            threat = self._general_threat_level(obs, *gen)
            if threat > 0 or phase == StrategicPhase.EMERGENCY_DEFENCE:
                reserve = 18
            elif phase in {StrategicPhase.OPENING, StrategicPhase.EXPANSION} and obs.turn < 120:
                reserve = 2
            elif obs.turn < 200:
                reserve = 6
            elif obs.turn < 600:
                reserve = 8
            else:
                reserve = 6
            gen_army = obs.army_grid[gr][gc]
            kept: list[Proposal] = []
            for p in proposals:
                a = p.action
                if a.kind == KIND_MOVE and (a.row, a.col) == (gr, gc):
                    sendable = gen_army - 1 if a.split == 0 else gen_army // 2
                    if gen_army - sendable < reserve:
                        continue
                kept.append(p)
            proposals = kept

        return proposals

    def _find_own_general(self, obs: Observation) -> tuple[int, int] | None:
        for r in range(obs.height):
            for c in range(obs.width):
                if obs.owner_grid[r][c] == OWNER_ME and obs.type_grid[r][c] == TYPE_GENERAL:
                    return r, c
        return None

    def _find_enemy_general(self, obs: Observation) -> tuple[int, int] | None:
        for r in range(obs.height):
            for c in range(obs.width):
                if obs.owner_grid[r][c] == OWNER_OPP and obs.type_grid[r][c] == TYPE_GENERAL:
                    return r, c
        return None

    def _has_enemy_visible(self, obs: Observation) -> bool:
        for r in range(obs.height):
            for c in range(obs.width):
                if obs.owner_grid[r][c] == OWNER_OPP:
                    return True
        return False

    def _general_threat_level(self, obs: Observation, gr: int, gc: int) -> int:
        """0=none, 1=enemy within 3, 2=enemy within 2, 3=adjacent large stack.

        Fog alone is not a threat — fog surrounds the general for most of the game.
        """
        level = 0
        for r in range(max(0, gr - 3), min(obs.height, gr + 4)):
            for c in range(max(0, gc - 3), min(obs.width, gc + 4)):
                dist = abs(r - gr) + abs(c - gc)
                if dist == 0 or dist > 3:
                    continue
                if obs.owner_grid[r][c] == OWNER_OPP:
                    level = max(
                        level,
                        3 if dist == 1 and obs.army_grid[r][c] >= 5 else (2 if dist <= 2 else 1),
                    )
        return level

    def _is_frontier(self, obs: Observation, r: int, c: int) -> bool:
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < obs.height and 0 <= nc < obs.width):
                continue
            if obs.owner_grid[nr][nc] != OWNER_ME:
                return True
        return False

    def _army_stats(self, obs: Observation) -> tuple[int, int, int, float]:
        mobile = 0
        frontier = 0
        stranded = 0
        total = 0
        for r in range(obs.height):
            for c in range(obs.width):
                if obs.owner_grid[r][c] != OWNER_ME:
                    continue
                a = obs.army_grid[r][c]
                total += a
                if a > 1:
                    mobile += a - 1
                    if self._is_frontier(obs, r, c):
                        frontier += a - 1
                    else:
                        stranded = max(stranded, a)
        conc = (frontier / mobile) if mobile > 0 else 0.0
        return mobile, frontier, stranded, conc
