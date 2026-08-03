"""Qualification-focused evaluation runner with explicit W/D/L metrics."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
from generals import GeneralsEnv
from generals.core import game

from generals_bot.evaluation.match import make_board, make_transition
from generals_bot.evaluation.qualification import (
    QualificationGameRecord,
    classify_expander_failure,
    outcome_from_winner,
    summarise_wdl,
)
from generals_bot.observation import GameContext
from generals_bot.policies.base import TraceLevel
from generals_bot.policies.phase_controller import is_dominant_position
from generals_bot.protocol import OWNER_OPP, TYPE_GENERAL
from generals_bot.selector import create_policy
from generals_bot.training.bridge_benchmark import extract_numpy_boards
from generals_bot.training.collect_bc import _action_to_jax, _observation_from_arrays

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_COMMIT = "9e3b9d13cca51caa1bb07db48bb85c9e90ce0462"


@dataclass(frozen=True)
class QualificationPreset:
    name: str
    seed_file: str
    seeds: int
    paired_positions: bool
    max_turns: int
    opponents: tuple[str, ...]
    label: str


PRESETS: dict[str, QualificationPreset] = {
    "qualification_smoke": QualificationPreset(
        name="qualification_smoke",
        seed_file="experiments/seeds/train.txt",
        seeds=8,
        paired_positions=True,
        max_turns=1200,
        opponents=("official_expander",),
        label="wiring_and_gross_regression",
    ),
    "qualification_development": QualificationPreset(
        name="qualification_development",
        seed_file="experiments/seeds/train.txt",
        seeds=48,
        paired_positions=True,
        max_turns=1200,
        opponents=("official_expander",),
        label="development",
    ),
    "qualification_holdout": QualificationPreset(
        name="qualification_holdout",
        seed_file="experiments/seeds/promotion_holdout.txt",
        seeds=48,
        paired_positions=True,
        max_turns=1200,
        opponents=("official_expander",),
        label="immutable_holdout",
    ),
    "qualification_hunter_info": QualificationPreset(
        name="qualification_hunter_info",
        seed_file="experiments/seeds/train.txt",
        seeds=12,
        paired_positions=True,
        max_turns=1200,
        opponents=("official_hunter",),
        label="informational_defence",
    ),
}


def _read_seeds(path: Path, n: int) -> list[int]:
    return [int(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()][:n]


def _play_qualification_game(
    policy_name: str,
    opponent: str,
    *,
    seed: int,
    swap: bool,
    max_turns: int,
) -> QualificationGameRecord:
    env = GeneralsEnv(mode="competition")
    transition = make_transition(env)
    get_obs = game.get_observation
    state = make_board(env, seed)
    h, w = (int(d) for d in state.armies.shape)
    names = [opponent, policy_name] if swap else [policy_name, opponent]
    p0 = create_policy(names[0], seed=seed)
    p1 = create_policy(names[1], seed=seed + 1)
    st0 = p0.initial_state(GameContext(0, h, w))
    st1 = p1.initial_state(GameContext(1, h, w))
    winner: int | None = None
    terminal_turn = 0
    first_contact: int | None = None
    eg_discovered = False
    turn_eg_discovered: int | None = None
    land_ratio = army_ratio = None
    remaining_enemy_land = None
    candidate_cells = None
    last_scout = None
    dominant = False
    diagnostics: dict[str, Any] = {}

    for turn_i in range(max_turns):
        eng0 = get_obs(state, 0)
        eng1 = get_obs(state, 1)
        t0, o0, a0, _, m0 = extract_numpy_boards(eng0, h, w)
        t1, o1, a1, _, m1 = extract_numpy_boards(eng1, h, w)
        obs0 = _observation_from_arrays(t0, o0, a0, m0)
        obs1 = _observation_from_arrays(t1, o1, a1, m1)
        # Track from candidate (policy_name) perspective
        cand_obs = obs1 if swap else obs0
        cand_state_holder = st1 if swap else st0
        if first_contact is None and any(
            cell == OWNER_OPP for row in cand_obs.owner_grid for cell in row
        ):
            first_contact = cand_obs.turn
        if not eg_discovered:
            for r in range(cand_obs.height):
                for c in range(cand_obs.width):
                    if (
                        cand_obs.owner_grid[r][c] == OWNER_OPP
                        and cand_obs.type_grid[r][c] == TYPE_GENERAL
                    ):
                        eg_discovered = True
                        turn_eg_discovered = cand_obs.turn
                        break
                if eg_discovered:
                    break

        try:
            d0 = p0.act(obs0, st0, deterministic=True, trace=TraceLevel.NONE, deadline=None)
            st0 = d0.new_state
            act0 = d0.action
        except Exception:
            from generals_bot.action import PASS_ACTION

            act0 = PASS_ACTION
        try:
            d1 = p1.act(obs1, st1, deterministic=True, trace=TraceLevel.NONE, deadline=None)
            st1 = d1.new_state
            act1 = d1.action
        except Exception:
            from generals_bot.action import PASS_ACTION

            act1 = PASS_ACTION

        state, info = transition(state, jnp.stack([_action_to_jax(act0), _action_to_jax(act1)]))
        terminal_turn = turn_i + 1
        if bool(info.is_done):
            winner = int(info.winner)
            break

    # Terminal telemetry from candidate perspective
    final_eng = get_obs(state, 1 if swap else 0)
    ft, fo, fa, _, fm = extract_numpy_boards(final_eng, h, w)
    final_obs = _observation_from_arrays(ft, fo, fa, fm)
    my_l, opp_l = final_obs.my_land, final_obs.opp_land
    my_a, opp_a = final_obs.my_army, final_obs.opp_army
    land_ratio = my_l / max(1, my_l + opp_l)
    army_ratio = my_a / max(1, my_a + opp_a)
    remaining_enemy_land = opp_l
    dominant = is_dominant_position(my_land=my_l, opp_land=opp_l)
    cand_policy_state = st1 if swap else st0
    if isinstance(cand_policy_state.data, dict):
        diagnostics = dict(cand_policy_state.data.get("diagnostics") or {})
        last_scout = cand_policy_state.data.get("last_newly_scouted_turn")
        candidate_cells = diagnostics.get("candidate_mask_size")

    perspective = 1 if swap else 0
    wins, draws, losses = outcome_from_winner(winner, perspective=perspective)
    if winner is None or (isinstance(winner, int) and winner < 0):
        terminal_reason = "DRAW_TURN_LIMIT"
    elif wins:
        terminal_reason = "WIN_GENERAL_CAPTURE"
    elif losses:
        terminal_reason = "LOSS_GENERAL_CAPTURED"
    else:
        terminal_reason = "DRAW_OTHER"

    record = QualificationGameRecord(
        policy=policy_name,
        opponent=opponent,
        seed=seed,
        position=perspective,
        winner=winner,
        terminal_turn=terminal_turn,
        terminal_reason=terminal_reason,
        wins=wins,
        draws=draws,
        losses=losses,
        first_enemy_contact_turn=first_contact,
        enemy_general_discovered=eg_discovered,
        turn_enemy_general_discovered=turn_eg_discovered,
        turn_enemy_general_captured=terminal_turn if wins else None,
        land_ratio_terminal=land_ratio,
        army_ratio_terminal=army_ratio,
        remaining_enemy_land=remaining_enemy_land,
        candidate_general_cells_terminal=candidate_cells,
        last_newly_scouted_turn=last_scout,
        dominant_at_terminal=dominant,
        extras=diagnostics,
    )
    if not wins:
        record.failure_class = classify_expander_failure(record)
    return record


def run_qualification_suite(
    *,
    policies: list[str],
    preset_name: str = "qualification_smoke",
    out_path: Path | None = None,
    wall_clock_s: float | None = None,
) -> dict[str, Any]:
    preset = PRESETS[preset_name]
    seed_path = REPO_ROOT / preset.seed_file
    seeds = _read_seeds(seed_path, preset.seeds)
    seed_hash = hashlib.sha256(seed_path.read_bytes()).hexdigest()
    t0 = time.perf_counter()
    records: list[QualificationGameRecord] = []
    stopped_early = False

    for policy in policies:
        for opponent in preset.opponents:
            for seed in seeds:
                swaps = (False, True) if preset.paired_positions else (False,)
                for swap in swaps:
                    if wall_clock_s is not None and (time.perf_counter() - t0) > wall_clock_s:
                        stopped_early = True
                        break
                    records.append(
                        _play_qualification_game(
                            policy,
                            opponent,
                            seed=seed,
                            swap=swap,
                            max_turns=preset.max_turns,
                        )
                    )
                if stopped_early:
                    break
            if stopped_early:
                break
        if stopped_early:
            break

    by_policy: dict[str, Any] = {}
    for policy in policies:
        subset = [r for r in records if r.policy == policy]
        by_policy[policy] = {
            "summary": summarise_wdl(subset),
            "by_opponent": {},
        }
        for opponent in preset.opponents:
            opp_recs = [r for r in subset if r.opponent == opponent]
            by_policy[policy]["by_opponent"][opponent] = summarise_wdl(opp_recs)

    report = {
        "schema_version": 1,
        "kind": "QUALIFICATION_SUITE",
        "preset": asdict(preset),
        "seed_manifest": preset.seed_file,
        "seed_manifest_hash": seed_hash,
        "engine_commit": ENGINE_COMMIT,
        "elapsed_s": time.perf_counter() - t0,
        "stopped_early": stopped_early,
        "games_count": len(records),
        "policies": by_policy,
        "games": [r.to_dict() for r in records],
        "note": "Do not use score_rate alone; report W/D/L and terminal reasons.",
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
