"""Corrected Stage 2 broader validation for Class A frozen INITIAL checkpoints."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from generals import GeneralsEnv
from generals.core import game

from generals_bot.action import PASS_ACTION
from generals_bot.evaluation.match import make_board, make_transition
from generals_bot.evaluation.qualification import score_rate
from generals_bot.observation import GameContext
from generals_bot.policies.base import TraceLevel
from generals_bot.selector import create_policy
from generals_bot.training.adaptive_initial import CheckpointPolicy
from generals_bot.training.bridge_benchmark import extract_numpy_boards
from generals_bot.training.collect_bc import _action_to_jax, _observation_from_arrays

REPO = Path(__file__).resolve().parents[1]
SEEDS = list(range(2000, 2004))  # 4 seeds × 2 seats × 3 opponents = 24 games/candidate
MAX_TURNS = 100
DEVICE = "cpu"
PORTAL = "heuristic_v2f_plus_planner_terminal_fix"
OPPONENTS = ["official_expander", "official_hunter", PORTAL]
OUT = REPO / "experiments" / "manifests" / "corrected_stage2_broader_validation.json"
DIAG = REPO / "replays" / "private" / "protocol_integrity"
LOG = REPO / "experiments" / "manifests" / "_stage2_run.log"


def _play_series(
    *,
    architecture: str,
    checkpoint: Path,
    opponent: str,
    seeds: list[int],
    learned_seat: int,
    max_turns: int,
    device: str,
    record_diagnostics: Path | None = None,
) -> dict[str, Any]:
    learned = CheckpointPolicy(architecture=architecture, checkpoint=checkpoint, device=device)
    opp = create_policy(opponent, seed=0)
    wins = draws = losses = 0
    faults = 0
    diagnostics_written = False
    for seed in seeds:
        env = GeneralsEnv(mode="competition")
        transition = make_transition(env)
        get_obs = game.get_observation
        state = make_board(env, seed)
        h, w = (int(d) for d in state.armies.shape)
        st_l = learned.initial_state(GameContext(learned_seat, h, w))
        st_o = opp.initial_state(GameContext(1 - learned_seat, h, w))
        winner = None
        frames: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        for turn in range(max_turns):
            eng0 = get_obs(state, 0)
            eng1 = get_obs(state, 1)
            t0, o0, a0, _, m0 = extract_numpy_boards(eng0, h, w)
            t1, o1, a1, _, m1 = extract_numpy_boards(eng1, h, w)
            obs0 = _observation_from_arrays(t0, o0, a0, m0)
            obs1 = _observation_from_arrays(t1, o1, a1, m1)
            try:
                if learned_seat == 0:
                    try:
                        d0 = learned.act(obs0, st_l, deterministic=True, trace=TraceLevel.NONE, deadline=None)
                        st_l = d0.new_state
                        act0 = d0.action
                    except Exception:
                        faults += 1
                        act0 = PASS_ACTION
                    try:
                        d1 = opp.act(obs1, st_o, deterministic=True, trace=TraceLevel.NONE, deadline=None)
                        st_o = d1.new_state
                        act1 = d1.action
                    except Exception:
                        faults += 1
                        act1 = PASS_ACTION
                else:
                    try:
                        d0 = opp.act(obs0, st_o, deterministic=True, trace=TraceLevel.NONE, deadline=None)
                        st_o = d0.new_state
                        act0 = d0.action
                    except Exception:
                        faults += 1
                        act0 = PASS_ACTION
                    try:
                        d1 = learned.act(obs1, st_l, deterministic=True, trace=TraceLevel.NONE, deadline=None)
                        st_l = d1.new_state
                        act1 = d1.action
                    except Exception:
                        faults += 1
                        act1 = PASS_ACTION
            except Exception:
                faults += 1
                act0 = PASS_ACTION
                act1 = PASS_ACTION
            if record_diagnostics is not None and not diagnostics_written:
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
                actions.append(
                    {
                        "turn": turn,
                        "act0": act0.as_tuple(),
                        "act1": act1.as_tuple(),
                        "learned_seat": learned_seat,
                    }
                )
            state, info = transition(state, jnp.stack([_action_to_jax(act0), _action_to_jax(act1)]))
            if bool(info.is_done):
                winner = int(info.winner)
                break
        if winner is None or winner < 0:
            draws += 1
            learned_result = "draw"
        elif winner == learned_seat:
            wins += 1
            learned_result = "win"
        else:
            losses += 1
            learned_result = "loss"
        if record_diagnostics is not None and not diagnostics_written and frames:
            record_diagnostics.parent.mkdir(parents=True, exist_ok=True)
            record_diagnostics.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "DIAGNOSTIC_REPLAY_FRAMES",
                        "frames_status": "RECORDED",
                        "architecture": architecture,
                        "checkpoint": str(checkpoint),
                        "seed": seed,
                        "learned_seat": learned_seat,
                        "opponent": opponent,
                        "winner": winner,
                        "learned_result": learned_result,
                        "turns": len(frames),
                        "frames": frames,
                        "actions": actions,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            diagnostics_written = True
    n = wins + draws + losses
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "games": n,
        "score_rate": score_rate(wins, draws, losses) if n else None,
        "protocol_faults": faults,
        "seeds": seeds,
        "learned_seat": learned_seat,
        "opponent": opponent,
        "diagnostics_path": str(record_diagnostics) if diagnostics_written and record_diagnostics else None,
    }


def main() -> None:
    LOG.write_text("", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    attribution = json.loads(
        (REPO / "experiments/manifests/portal_attribution_gate.json").read_text(encoding="utf-8")
    )
    if attribution.get("decision") != "RESOLVED":
        raise SystemExit(f"PORTAL_ATTRIBUTION_GATE={attribution.get('decision')}; Stage 2 portal branch blocked")

    candidates = [
        {
            "arm_id": "cnn_bc_init_seed11",
            "architecture": "recurrent_cnn_v2",
            "checkpoint": REPO
            / "experiments/checkpoints/initial/initial_cnn_cnn_bc_init_seed11/chunk_0/model.json",
        },
        {
            "arm_id": "graph_bc_init_seed7",
            "architecture": "recurrent_graph_belief_v2",
            "checkpoint": REPO
            / "experiments/checkpoints/initial/initial_graph_graph_bc_init_seed7/chunk_0/model.json",
        },
    ]

    results = []
    for cand in candidates:
        ckpt = cand["checkpoint"]
        sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        series = []
        for opp in OPPONENTS:
            for seat in (0, 1):
                # Record diagnostics only for first series of each architecture (evidence sample).
                diag = None
                if opp == OPPONENTS[0] and seat == 0:
                    diag = DIAG / f"stage2_{cand['arm_id']}_{opp}_seat{seat}_s{SEEDS[0]}.json"
                log(f"STAGE2 {cand['arm_id']} vs {opp} seat={seat}")
                res = _play_series(
                    architecture=cand["architecture"],
                    checkpoint=ckpt,
                    opponent=opp,
                    seeds=SEEDS,
                    learned_seat=seat,
                    max_turns=MAX_TURNS,
                    device=DEVICE,
                    record_diagnostics=diag,
                )
                log(json.dumps({k: res[k] for k in res if k != "diagnostics_path"}))
                series.append(res)
        wins = sum(s["wins"] for s in series)
        draws = sum(s["draws"] for s in series)
        losses = sum(s["losses"] for s in series)
        faults = sum(s["protocol_faults"] for s in series)
        games = wins + draws + losses
        results.append(
            {
                "arm_id": cand["arm_id"],
                "architecture": cand["architecture"],
                "checkpoint": str(ckpt),
                "checkpoint_sha256": sha,
                "checkpoint_sha256_16": sha[:16],
                "frozen_initial_best": True,
                "portal_candidate": PORTAL,
                "portal_attribution": "RESOLVED",
                "by_series": series,
                "aggregate": {
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "games": games,
                    "score_rate": (wins + 0.5 * draws) / games if games else None,
                    "protocol_faults": faults,
                    "seeds": SEEDS,
                    "positions": [0, 1],
                    "opponents": OPPONENTS,
                },
            }
        )

    payload = {
        "schema_version": 1,
        "kind": "CORRECTED_STAGE2_BROADER_VALIDATION",
        "gate_name": "CORRECTED_STAGE2_BROADER_VALIDATION",
        "research_generation_id": "protocol_dashboard_integrity_2026-08-04",
        "decision": "COMPLETE",
        "reasons": [
            "PORTAL_ATTRIBUTION_GATE=RESOLVED",
            f"Opponents: {OPPONENTS}",
            "4 seeds × 2 seats × 3 opponents = 24 games per architecture",
        ],
        "blockers": [],
        "candidates": results,
        "engine_sha": "9e3b9d13cca51caa1bb07db48bb85c9e90ce0462",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "supersedes": None,
        "superseded_by": None,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log(f"WROTE {OUT}")


if __name__ == "__main__":
    main()
