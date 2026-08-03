"""Deterministic evaluation: run the same matchup twice and find first divergence."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
from generals import GeneralsEnv
from generals.core import game

from generals_bot.evaluation.match import make_board, make_transition
from generals_bot.observation import GameContext
from generals_bot.policies.base import TraceLevel
from generals_bot.selector import create_policy
from generals_bot.training.bridge_benchmark import extract_numpy_boards
from generals_bot.training.collect_bc import _action_to_jax, _observation_from_arrays


def seed_all(seed: int) -> None:
    """Seed Python / NumPy / Torch / JAX consumer RNGs when present."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001
        pass


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _obs_hash(obs) -> str:
    payload = (
        f"{obs.turn}|{obs.height}x{obs.width}|"
        f"{obs.my_land},{obs.my_army},{obs.opp_land},{obs.opp_army}|"
        f"{obs.type_grid}|{obs.owner_grid}|{obs.army_grid}"
    )
    return _hash_bytes(payload.encode("utf-8"))


def _action_tuple(action) -> tuple:
    return (int(action.kind), int(action.row), int(action.col), int(action.direction), int(action.split))


def _state_hash(state) -> str:
    # PolicyState.data may hold MapMemory etc. — hash a stable subset.
    data = state.data if isinstance(state.data, dict) else {}
    keys = sorted(k for k in data.keys() if k not in {"memory", "diagnostics", "threat", "scout_task"})
    parts = [f"phase={data.get('phase')}", f"reason={data.get('phase_reason')}"]
    for k in keys:
        v = data[k]
        if isinstance(v, (str, int, float, bool, type(None))):
            parts.append(f"{k}={v}")
        elif isinstance(v, tuple) and all(isinstance(x, (int, float)) for x in v):
            parts.append(f"{k}={v}")
    mem = data.get("memory")
    if mem is not None and hasattr(mem, "ever_seen"):
        seen = sum(1 for row in mem.ever_seen for x in row if x)
        parts.append(f"seen={seen}")
    return _hash_bytes("|".join(parts).encode("utf-8"))


@dataclass
class TurnTrace:
    turn: int
    obs_hash: str
    state_hash: str
    phase: str | None
    action: tuple
    option: str | None
    opponent_action: tuple | None
    board_hash: str


def play_traced_game(
    policy_name: str,
    opponent: str,
    *,
    seed: int,
    swap: bool,
    max_turns: int = 1200,
    rng_seed: int | None = None,
) -> dict[str, Any]:
    """Play one game and return per-turn traces plus terminal summary."""
    seed_all(rng_seed if rng_seed is not None else seed)
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
    traces: list[dict[str, Any]] = []
    cand_idx = 1 if swap else 0

    for turn_i in range(max_turns):
        eng0 = get_obs(state, 0)
        eng1 = get_obs(state, 1)
        t0, o0, a0, _, m0 = extract_numpy_boards(eng0, h, w)
        t1, o1, a1, _, m1 = extract_numpy_boards(eng1, h, w)
        obs0 = _observation_from_arrays(t0, o0, a0, m0)
        obs1 = _observation_from_arrays(t1, o1, a1, m1)
        board_hash = _hash_bytes(
            (
                str(jnp.asarray(state.armies).tolist())
                + "|"
                + str(jnp.asarray(state.ownership).tolist())
                + "|"
                + str(turn_i)
            ).encode("utf-8")
        )

        d0 = p0.act(obs0, st0, deterministic=True, trace=TraceLevel.NONE, deadline=None)
        st0 = d0.new_state
        d1 = p1.act(obs1, st1, deterministic=True, trace=TraceLevel.NONE, deadline=None)
        st1 = d1.new_state

        cand_dec = d1 if swap else d0
        cand_obs = obs1 if swap else obs0
        cand_st = st1 if swap else st0
        opp_dec = d0 if swap else d1

        traces.append(
            asdict(
                TurnTrace(
                    turn=cand_obs.turn,
                    obs_hash=_obs_hash(cand_obs),
                    state_hash=_state_hash(cand_st),
                    phase=(cand_st.data.get("phase") if isinstance(cand_st.data, dict) else None),
                    action=_action_tuple(cand_dec.action),
                    option=cand_dec.strategic_option,
                    opponent_action=_action_tuple(opp_dec.action),
                    board_hash=board_hash,
                )
            )
        )

        state, info = transition(state, jnp.stack([_action_to_jax(d0.action), _action_to_jax(d1.action)]))
        if bool(info.is_done):
            winner = int(info.winner)
            break

    perspective = cand_idx
    if winner is None:
        wdl = (0, 1, 0)
    elif winner == perspective:
        wdl = (1, 0, 0)
    else:
        wdl = (0, 0, 1)

    return {
        "policy": policy_name,
        "opponent": opponent,
        "seed": seed,
        "swap": swap,
        "height": h,
        "width": w,
        "winner": winner,
        "terminal_turn": len(traces),
        "wins": wdl[0],
        "draws": wdl[1],
        "losses": wdl[2],
        "traces": traces,
    }


def first_divergence(a: dict, b: dict) -> dict[str, Any] | None:
    ta, tb = a["traces"], b["traces"]
    n = min(len(ta), len(tb))
    for i in range(n):
        for key in ("board_hash", "obs_hash", "action", "option", "opponent_action", "phase", "state_hash"):
            if ta[i].get(key) != tb[i].get(key):
                return {
                    "index": i,
                    "turn": ta[i].get("turn"),
                    "field": key,
                    "a": ta[i],
                    "b": tb[i],
                }
    if len(ta) != len(tb):
        return {"index": n, "turn": None, "field": "trace_length", "a": len(ta), "b": len(tb)}
    if (a["winner"], a["terminal_turn"]) != (b["winner"], b["terminal_turn"]):
        return {
            "index": n,
            "turn": None,
            "field": "terminal",
            "a": {"winner": a["winner"], "terminal_turn": a["terminal_turn"]},
            "b": {"winner": b["winner"], "terminal_turn": b["terminal_turn"]},
        }
    return None


def compare_same_process(
    policy_name: str,
    opponent: str,
    *,
    seed: int,
    swap: bool,
    max_turns: int = 1200,
) -> dict[str, Any]:
    a = play_traced_game(policy_name, opponent, seed=seed, swap=swap, max_turns=max_turns)
    b = play_traced_game(policy_name, opponent, seed=seed, swap=swap, max_turns=max_turns)
    div = first_divergence(a, b)
    return {
        "mode": "same_process",
        "policy": policy_name,
        "opponent": opponent,
        "seed": seed,
        "swap": swap,
        "identical": div is None,
        "divergence": div,
        "a_terminal": {"winner": a["winner"], "turn": a["terminal_turn"], "wdl": (a["wins"], a["draws"], a["losses"])},
        "b_terminal": {"winner": b["winner"], "turn": b["terminal_turn"], "wdl": (b["wins"], b["draws"], b["losses"])},
        "board": {"h": a["height"], "w": a["width"]},
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="heuristic_v2f_best_reference")
    p.add_argument("--opponent", default="official_expander")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--swap", action="store_true")
    p.add_argument("--max-turns", type=int, default=1200)
    p.add_argument("--out", type=Path, default=Path("experiments/manifests/determinism_probe.json"))
    args = p.parse_args()
    result = compare_same_process(
        args.policy, args.opponent, seed=args.seed, swap=args.swap, max_turns=args.max_turns
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "divergence"}, indent=2))
    if result["divergence"]:
        print("DIVERGENCE", json.dumps(result["divergence"], indent=2)[:2000])
    else:
        print("IDENTICAL")


if __name__ == "__main__":
    main()
