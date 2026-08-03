"""Turn-level diagnostics for Expander draws (Phase 9Q)."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
from generals import GeneralsEnv
from generals.core import game

from generals_bot.evaluation.match import make_board, make_transition
from generals_bot.observation import GameContext
from generals_bot.policies.base import TraceLevel
from generals_bot.protocol import OWNER_OPP, TYPE_GENERAL
from generals_bot.selector import create_policy
from generals_bot.training.bridge_benchmark import extract_numpy_boards
from generals_bot.training.collect_bc import _action_to_jax, _observation_from_arrays

REPO_ROOT = Path(__file__).resolve().parents[3]


def diagnose_game(policy_name: str, opponent: str, seed: int, swap: bool, max_turns: int = 1200) -> dict:
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
    timeline = []
    first_contact = None
    last_contact = None
    eg_turn = None
    emergency_events = []
    winner = None
    terminal_turn = 0

    for turn_i in range(max_turns):
        eng0 = get_obs(state, 0)
        eng1 = get_obs(state, 1)
        t0, o0, a0, _, m0 = extract_numpy_boards(eng0, h, w)
        t1, o1, a1, _, m1 = extract_numpy_boards(eng1, h, w)
        obs0 = _observation_from_arrays(t0, o0, a0, m0)
        obs1 = _observation_from_arrays(t1, o1, a1, m1)
        cand_obs = obs1 if swap else obs0
        visible = any(cell == OWNER_OPP for row in cand_obs.owner_grid for cell in row)
        if visible:
            if first_contact is None:
                first_contact = cand_obs.turn
            last_contact = cand_obs.turn
        eg = False
        for r in range(cand_obs.height):
            for c in range(cand_obs.width):
                if cand_obs.owner_grid[r][c] == OWNER_OPP and cand_obs.type_grid[r][c] == TYPE_GENERAL:
                    eg = True
                    if eg_turn is None:
                        eg_turn = cand_obs.turn
        d0 = p0.act(obs0, st0, deterministic=True, trace=TraceLevel.NONE, deadline=None)
        st0 = d0.new_state
        d1 = p1.act(obs1, st1, deterministic=True, trace=TraceLevel.NONE, deadline=None)
        st1 = d1.new_state
        cand_dec = d1 if swap else d0
        cand_st = st1 if swap else st0
        diag = dict(cand_st.data.get("diagnostics") or {})
        if turn_i % 50 == 0 or diag.get("phase") in {
            "EMERGENCY_DEFENCE",
            "DRAW_AVOIDANCE",
            "DEATHTOUCH_HUNT",
            "CLEAR_ENEMY_REGION",
            "SEARCH_FOR_CONTACT",
        }:
            timeline.append(
                {
                    "turn": cand_obs.turn,
                    "phase": diag.get("phase"),
                    "option": cand_dec.strategic_option,
                    "scout_target": diag.get("scout_target"),
                    "mask": diag.get("candidate_mask_size"),
                    "regions": diag.get("unresolved_regions"),
                    "emergency": diag.get("emergency_active"),
                    "caution": diag.get("caution_active"),
                    "trigger": diag.get("threat_trigger"),
                    "reserve": diag.get("general_reserve"),
                    "module": (cand_dec.shield_result or {}).get("selected_module"),
                }
            )
        if diag.get("emergency_active") and (
            not emergency_events or emergency_events[-1].get("end") is not None
        ):
            emergency_events.append(
                {
                    "start": cand_obs.turn,
                    "trigger": diag.get("threat_trigger"),
                    "end": None,
                }
            )
        elif emergency_events and emergency_events[-1].get("end") is None and not diag.get(
            "emergency_active"
        ):
            emergency_events[-1]["end"] = cand_obs.turn
            emergency_events[-1]["exit"] = diag.get("threat_exit_reason")

        state, info = transition(
            state, jnp.stack([_action_to_jax(d0.action), _action_to_jax(d1.action)])
        )
        terminal_turn = turn_i + 1
        if bool(info.is_done):
            winner = int(info.winner)
            break

    final_diag = dict((st1 if swap else st0).data.get("diagnostics") or {})
    subclass = "unknown"
    if first_contact is None:
        subclass = "never_contacted_opponent"
    elif eg_turn is None and last_contact is not None and last_contact < terminal_turn - 100:
        subclass = "contacted_then_lost_opponent"
    elif eg_turn is None:
        subclass = "enemy_territory_found_but_not_cleared"
    elif winner is None or winner < 0:
        subclass = "candidate_general_known_but_not_converted"

    return {
        "seed": seed,
        "swap": swap,
        "policy": policy_name,
        "opponent": opponent,
        "winner": winner,
        "terminal_turn": terminal_turn,
        "first_enemy_contact_turn": first_contact,
        "last_enemy_contact_turn": last_contact,
        "enemy_general_discovered": eg_turn is not None,
        "general_discovery_turn": eg_turn,
        "emergency_events": emergency_events,
        "timeline_sample": timeline,
        "terminal_diagnostics": final_diag,
        "draw_subclass": subclass,
    }


def main() -> None:
    # Latest smoke draws from v2j: seeds with draws
    draws = [
        (0, False),
        (0, True),
        (1, False),
        (1, True),
        (2, True),
        (3, True),
        (4, False),
        (4, True),
        (5, False),
        (5, True),
        (6, True),
        (7, False),
    ]
    reports = []
    for seed, swap in draws:
        print(f"diagnosing seed={seed} swap={swap}")
        reports.append(
            diagnose_game("heuristic_v2_qualifier", "official_expander", seed, swap)
        )
    out = REPO_ROOT / "experiments" / "manifests" / "expander_draw_diagnostics_9q.json"
    summary = {
        "schema_version": 1,
        "kind": "EXPANDER_DRAW_DIAGNOSTICS",
        "games": reports,
        "subclass_counts": {},
    }
    for r in reports:
        key = r["draw_subclass"]
        summary["subclass_counts"][key] = summary["subclass_counts"].get(key, 0) + 1
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["subclass_counts"], indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
