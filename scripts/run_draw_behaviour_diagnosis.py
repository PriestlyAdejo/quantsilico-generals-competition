"""Phase 9D Stage A — draw behaviour diagnosis for frozen CNN and graph checkpoints."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from generals import GeneralsEnv
from generals.core import game

from generals_bot.action import KIND_MOVE, KIND_PASS, PASS_ACTION
from generals_bot.evaluation.match import make_board, make_transition
from generals_bot.observation import GameContext
from generals_bot.policies.base import TraceLevel
from generals_bot.selector import create_policy
from generals_bot.training.adaptive_initial import CheckpointPolicy
from generals_bot.training.bridge_benchmark import extract_numpy_boards
from generals_bot.training.collect_bc import _action_to_jax, _observation_from_arrays

REPO = Path(__file__).resolve().parents[1]
SEEDS_MANIFEST = REPO / "experiments/manifests/seeds/phase9d_seed_partitions.json"
DIAG_DIR = REPO / "replays/private/protocol_integrity"
OUT_DIR = REPO / "experiments/manifests"
MAX_TURNS = 100
DEVICE = "cpu"


def _inverse_move(a0: tuple, a1: tuple) -> bool:
    if a0[0] != KIND_MOVE or a1[0] != KIND_MOVE:
        return False
    # (kind,r,c,dir,split) — reverse direction heuristic: dir xor 1 for cardinal pairs 0↔1, 2↔3
    return a0[1] == a1[1] and a0[2] == a1[2] and ((a0[3] ^ 1) == a1[3])


def play_diagnosed_game(
    *,
    architecture: str,
    checkpoint: Path,
    seed: int,
    learned_seat: int,
    max_turns: int,
    record_path: Path | None,
) -> dict[str, Any]:
    learned = CheckpointPolicy(architecture=architecture, checkpoint=checkpoint, device=DEVICE)
    opp = create_policy("official_expander", seed=0)
    env = GeneralsEnv(mode="competition")
    transition = make_transition(env)
    get_obs = game.get_observation
    state = make_board(env, seed)
    h, w = (int(d) for d in state.armies.shape)
    st_l = learned.initial_state(GameContext(learned_seat, h, w))
    st_o = opp.initial_state(GameContext(1 - learned_seat, h, w))

    faults = 0
    fallbacks = 0
    pass_count = 0
    oscillation = 0
    first_legal_move_turn = None
    first_contact_turn = None
    first_enemy_owned_turn = None
    discovery_turn = None
    turns_after_discovery_no_attack = 0
    attack_attempts = 0
    prev_learned_act: tuple | None = None
    frames: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    winner = None
    discovered = False

    for turn in range(max_turns):
        eng0 = get_obs(state, 0)
        eng1 = get_obs(state, 1)
        t0, o0, a0, _, m0 = extract_numpy_boards(eng0, h, w)
        t1, o1, a1, _, m1 = extract_numpy_boards(eng1, h, w)
        obs0 = _observation_from_arrays(t0, o0, a0, m0)
        obs1 = _observation_from_arrays(t1, o1, a1, m1)

        # Policy-visible contact / discovery from learned seat observation
        learned_obs = obs0 if learned_seat == 0 else obs1
        owner = np.asarray(learned_obs.owner_grid)
        enemy_visible = int(np.sum(owner == 2))  # OWNER_OPP convention in protocol
        # generals_bot protocol OWNER_OPP — check import
        from generals_bot.protocol import OWNER_OPP, TYPE_GENERAL

        enemy_visible = int(np.sum(np.asarray(learned_obs.owner_grid) == OWNER_OPP))
        enemy_general = int(
            np.sum(
                (np.asarray(learned_obs.type_grid) == TYPE_GENERAL)
                & (np.asarray(learned_obs.owner_grid) == OWNER_OPP)
            )
        )
        if enemy_visible > 0 and first_contact_turn is None:
            first_contact_turn = turn
            first_enemy_owned_turn = turn
        if enemy_general > 0 and discovery_turn is None:
            discovery_turn = turn
            discovered = True

        try:
            if learned_seat == 0:
                d0 = learned.act(obs0, st_l, deterministic=True, trace=TraceLevel.NONE, deadline=None)
                st_l = d0.new_state
                act0 = d0.action
                d1 = opp.act(obs1, st_o, deterministic=True, trace=TraceLevel.NONE, deadline=None)
                st_o = d1.new_state
                act1 = d1.action
            else:
                d0 = opp.act(obs0, st_o, deterministic=True, trace=TraceLevel.NONE, deadline=None)
                st_o = d0.new_state
                act0 = d0.action
                d1 = learned.act(obs1, st_l, deterministic=True, trace=TraceLevel.NONE, deadline=None)
                st_l = d1.new_state
                act1 = d1.action
        except Exception:
            faults += 1
            fallbacks += 1
            act0 = PASS_ACTION
            act1 = PASS_ACTION

        learned_act = act0 if learned_seat == 0 else act1
        la = learned_act.as_tuple()
        if la[0] == KIND_PASS:
            pass_count += 1
        else:
            if first_legal_move_turn is None:
                first_legal_move_turn = turn
        if prev_learned_act is not None and _inverse_move(prev_learned_act, la):
            oscillation += 1
        if la[0] == KIND_MOVE and discovered:
            attack_attempts += 1
        elif discovered and la[0] == KIND_PASS:
            turns_after_discovery_no_attack += 1
        prev_learned_act = la

        frames.append(
            {
                "turn": turn,
                "height": h,
                "width": w,
                "type_grid": np.asarray(t0).tolist(),
                "owner_grid": np.asarray(o0).tolist(),
                "army_grid": np.asarray(a0).tolist(),
            }
        )
        actions.append({"turn": turn, "act0": act0.as_tuple(), "act1": act1.as_tuple(), "learned_seat": learned_seat})

        state, info = transition(state, jnp.stack([_action_to_jax(act0), _action_to_jax(act1)]))
        if bool(info.is_done):
            winner = int(info.winner)
            break

    # End-state stats from learned perspective
    final_eng = get_obs(state, learned_seat)
    ft, fo, fa, _, _ = extract_numpy_boards(final_eng, h, w)
    from generals_bot.protocol import OWNER_ME

    army_cap = int(np.sum(np.asarray(fa)[np.asarray(fo) == OWNER_ME]))
    territory_cap = int(np.sum(np.asarray(fo) == OWNER_ME))
    turns_played = len(actions)

    if winner is None or winner < 0:
        result = "draw"
        terminal_reason = "TURN_CAP" if turns_played >= max_turns else "DRAW"
    elif winner == learned_seat:
        result = "win"
        terminal_reason = "WIN"
    else:
        result = "loss"
        terminal_reason = "LOSS"

    if record_path is not None:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "DIAGNOSTIC_REPLAY_FRAMES",
                    "frames_status": "RECORDED",
                    "architecture": architecture,
                    "checkpoint": str(checkpoint),
                    "seed": seed,
                    "learned_seat": learned_seat,
                    "opponent": "official_expander",
                    "winner": winner,
                    "learned_result": result,
                    "turns": turns_played,
                    "frames": frames,
                    "actions": actions,
                    "events": [
                        {"type": "first_contact", "turn": first_contact_turn},
                        {"type": "discovery", "turn": discovery_turn},
                        {"type": "terminal", "reason": terminal_reason},
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )

    return {
        "seed": seed,
        "learned_seat": learned_seat,
        "result": result,
        "terminal_reason": terminal_reason,
        "turns": turns_played,
        "protocol_faults": faults,
        "fallback_actions": fallbacks,
        "pass_count": pass_count,
        "pass_rate": pass_count / max(turns_played, 1),
        "oscillation_count": oscillation,
        "first_legal_move_turn": first_legal_move_turn,
        "first_contact_turn": first_contact_turn,
        "first_enemy_owned_turn": first_enemy_owned_turn,
        "discovery_turn": discovery_turn,
        "turns_after_discovery_without_attack": turns_after_discovery_no_attack,
        "attack_attempts": attack_attempts,
        "army_at_turn_cap": army_cap,
        "territory_at_turn_cap": territory_cap,
        "diagnostics_path": str(record_path) if record_path else None,
    }


def classify(games: list[dict[str, Any]]) -> dict[str, Any]:
    mean_pass = float(np.mean([g["pass_rate"] for g in games]))
    mean_osc = float(np.mean([g["oscillation_count"] for g in games]))
    discoveries = [g["discovery_turn"] for g in games if g["discovery_turn"] is not None]
    contacts = [g["first_contact_turn"] for g in games if g["first_contact_turn"] is not None]
    attacks = float(np.mean([g["attack_attempts"] for g in games]))
    post_disc_idle = float(np.mean([g["turns_after_discovery_without_attack"] for g in games]))

    labels: list[str] = []
    if mean_pass >= 0.55:
        labels.append("PASSIVITY")
    if mean_osc >= 8:
        labels.append("OSCILLATION")
    if len(contacts) < len(games) // 2:
        labels.append("CONTACT_FAILURE")
    if len(discoveries) == 0:
        labels.append("DISCOVERY_FAILURE")
    elif attacks < 1.0 and post_disc_idle >= 10:
        labels.append("ATTACK_INITIATION_FAILURE")
    elif attacks >= 1.0 and all(g["result"] != "win" for g in games):
        labels.append("CONVERSION_FAILURE")

    # Seat consistency
    by_seat = {0: [], 1: []}
    for g in games:
        by_seat[g["learned_seat"]].append(g["pass_rate"])
    seat_gap = abs(float(np.mean(by_seat[0] or [0])) - float(np.mean(by_seat[1] or [0])))

    if not labels:
        dominant = "INSUFFICIENT_EVIDENCE"
        secondary = None
        sufficiency = "INSUFFICIENT_EVIDENCE"
    elif len(set(labels)) > 2 and seat_gap > 0.25:
        dominant = "INSUFFICIENT_EVIDENCE"
        secondary = labels[0]
        sufficiency = "INSUFFICIENT_EVIDENCE"
    else:
        # Prefer first diagnostic priority
        priority = [
            "PASSIVITY",
            "OSCILLATION",
            "CONTACT_FAILURE",
            "DISCOVERY_FAILURE",
            "ATTACK_INITIATION_FAILURE",
            "CONVERSION_FAILURE",
        ]
        ordered = [p for p in priority if p in labels]
        dominant = ordered[0]
        secondary = ordered[1] if len(ordered) > 1 else None
        sufficiency = "SUFFICIENT" if len(games) >= 4 else "MARGINAL"

    return {
        "aggregate": {
            "games": len(games),
            "mean_pass_rate": mean_pass,
            "mean_oscillation": mean_osc,
            "discovery_rate": len(discoveries) / max(len(games), 1),
            "contact_rate": len(contacts) / max(len(games), 1),
            "mean_attack_attempts": attacks,
            "mean_post_discovery_idle": post_disc_idle,
            "seat_pass_rate_gap": seat_gap,
            "results": Counter(g["result"] for g in games),
        },
        "dominant_diagnosis": dominant,
        "secondary_diagnosis": secondary,
        "candidate_labels": labels,
        "evidence_sufficiency": sufficiency,
    }


def main() -> None:
    seeds_doc = json.loads(SEEDS_MANIFEST.read_text(encoding="utf-8"))
    seeds = list(seeds_doc["partitions"]["diagnosis_seeds"]["seeds"])
    candidates = [
        {
            "arm_id": "cnn_bc_init_seed11",
            "architecture": "recurrent_cnn_v2",
            "checkpoint": REPO
            / "experiments/checkpoints/initial/initial_cnn_cnn_bc_init_seed11/chunk_0/model.json",
            "out": OUT_DIR / "draw_behaviour_diagnosis_cnn.json",
        },
        {
            "arm_id": "graph_bc_init_seed7",
            "architecture": "recurrent_graph_belief_v2",
            "checkpoint": REPO
            / "experiments/checkpoints/initial/initial_graph_graph_bc_init_seed7/chunk_0/model.json",
            "out": OUT_DIR / "draw_behaviour_diagnosis_graph.json",
        },
    ]
    for cand in candidates:
        ckpt = cand["checkpoint"]
        sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        games = []
        print(f"DIAGNOSE {cand['arm_id']}", flush=True)
        for seed in seeds:
            for seat in (0, 1):
                path = DIAG_DIR / f"phase9d_diag_{cand['arm_id']}_s{seed}_seat{seat}.json"
                g = play_diagnosed_game(
                    architecture=cand["architecture"],
                    checkpoint=ckpt,
                    seed=seed,
                    learned_seat=seat,
                    max_turns=MAX_TURNS,
                    record_path=path,
                )
                print(json.dumps({k: g[k] for k in g if k != "diagnostics_path"}), flush=True)
                games.append(g)
        # Expand if dominant would be inconsistent — handled in classify via sufficiency
        summary = classify(games)
        # Expand to 8 games if INSUFFICIENT and we have expansion seeds listed
        if summary["evidence_sufficiency"] == "INSUFFICIENT_EVIDENCE":
            extra = [2102, 2103]
            for seed in extra:
                for seat in (0, 1):
                    path = DIAG_DIR / f"phase9d_diag_{cand['arm_id']}_s{seed}_seat{seat}.json"
                    g = play_diagnosed_game(
                        architecture=cand["architecture"],
                        checkpoint=ckpt,
                        seed=seed,
                        learned_seat=seat,
                        max_turns=MAX_TURNS,
                        record_path=path,
                    )
                    games.append(g)
            summary = classify(games)

        payload = {
            "schema_version": 1,
            "kind": "DRAW_BEHAVIOUR_DIAGNOSIS",
            "gate_name": "DRAW_BEHAVIOUR_DIAGNOSIS",
            "research_generation_id": "phase9d_draw_conversion_2026-08-04",
            "arm_id": cand["arm_id"],
            "architecture": cand["architecture"],
            "checkpoint": str(ckpt),
            "checkpoint_sha256": sha,
            "opponent": "official_expander",
            "seeds": sorted({g["seed"] for g in games}),
            "games": games,
            **summary,
            "decision": summary["dominant_diagnosis"],
            "reasons": [
                f"dominant={summary['dominant_diagnosis']}",
                f"sufficiency={summary['evidence_sufficiency']}",
            ],
            "blockers": [],
            "engine_sha": "9e3b9d13cca51caa1bb07db48bb85c9e90ce0462",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "supersedes": None,
            "superseded_by": None,
        }
        # JSON-ify Counter
        payload["aggregate"]["results"] = dict(payload["aggregate"]["results"])
        cand["out"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("WROTE", cand["out"], "dominant", summary["dominant_diagnosis"], flush=True)


if __name__ == "__main__":
    main()
