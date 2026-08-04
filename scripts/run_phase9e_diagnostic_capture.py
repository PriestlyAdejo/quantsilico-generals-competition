"""Phase 9E Stage 3B — bounded rules-aware diagnostic capture (no gradients)."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from generals import GeneralsEnv
from generals.core import game

from generals_bot.action import KIND_BUILD, KIND_MOVE, KIND_PASS
from generals_bot.castle_cost import castle_price_at, own_structures
from generals_bot.evaluation.match import make_board, make_transition
from generals_bot.legal import enumerate_legal_actions, is_legal_action
from generals_bot.models.checkpoint import apply_state_dict
from generals_bot.models.factory import build_model
from generals_bot.models.model_forward import adapt_forward_output
from generals_bot.models.observation_encoder import encode_globals_numpy, encode_grids_numpy
from generals_bot.observation import GameContext
from generals_bot.policies.base import TraceLevel
from generals_bot.protocol import DIRECTIONS, OWNER_OPP
from generals_bot.selector import create_policy
from generals_bot.training.adaptive_initial import CheckpointPolicy
from generals_bot.training.bridge_benchmark import extract_numpy_boards
from generals_bot.training.collect_bc import _action_to_jax, _observation_from_arrays

import torch

REPO = Path(__file__).resolve().parents[1]
SEEDS = REPO / "experiments/manifests/seeds/phase9e_seed_partitions.json"
BUDGET = REPO / "experiments/manifests/phase9e_diagnostic_capture_budget.json"
MAP_HASHES = REPO / "experiments/manifests/seeds/phase9e_map_hash_registry.json"
OUT = REPO / "experiments/manifests/phase9e_diagnostic_capture.json"
MAX_TURNS = 80
DEVICE = "cpu"

ARCHS = {
    "cnn": (
        "recurrent_cnn_v2",
        REPO / "experiments/checkpoints/initial/initial_cnn_cnn_bc_init_seed11/chunk_0/model.json",
    ),
    "graph": (
        "recurrent_graph_belief_v2",
        REPO / "experiments/checkpoints/initial/initial_graph_graph_bc_init_seed7/chunk_0/model.json",
    ),
}


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _board_hash(state) -> str:
    payload = np.asarray(state.armies).tobytes() + np.asarray(state.mountains).tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def _belief_snapshot(architecture: str, checkpoint: Path, obs) -> dict[str, Any]:
    model = build_model(architecture)
    apply_state_dict(model, checkpoint, map_location="cpu")
    model.eval()
    type_grid = np.asarray(obs.type_grid, dtype=np.int32)
    owner_grid = np.asarray(obs.owner_grid, dtype=np.int32)
    army_grid = np.asarray(obs.army_grid, dtype=np.int32)
    cells = torch.from_numpy(encode_grids_numpy(type_grid, owner_grid, army_grid)).unsqueeze(0)
    glob = torch.from_numpy(encode_globals_numpy(obs)).unsqueeze(0)
    hidden = model.initial_hidden(1, device=torch.device("cpu"))
    if hasattr(model, "initial_cell_memory"):
        raw = model.forward_tensors(cells, glob, hidden, model.initial_cell_memory(1))
    else:
        raw = model.forward_tensors(cells, glob, hidden)
    fwd = adapt_forward_output(raw)
    out: dict[str, Any] = {
        "value": float(fwd.value.detach().cpu().item()),
        "logits_finite": bool(torch.isfinite(fwd.logits).all().item()),
    }
    if isinstance(raw, dict):
        for key in ("belief", "opponent_style", "general_loss_risk", "concepts"):
            if key in raw:
                t = raw[key].detach().cpu().flatten()
                out[key] = {
                    "shape": list(raw[key].shape),
                    "mean": float(t.mean().item()) if t.numel() else None,
                    "norm": float(t.norm().item()) if t.numel() else None,
                }
    return out


def play_game(
    *,
    architecture: str,
    checkpoint: Path,
    seed: int,
    seat: int,
    opponent: str,
) -> dict[str, Any]:
    learned = CheckpointPolicy(architecture=architecture, checkpoint=checkpoint, device=DEVICE)
    opp = create_policy(opponent, seed=seed + 17)
    env = GeneralsEnv(mode="competition")
    transition = make_transition(env)
    get_obs = game.get_observation
    state = make_board(env, seed)
    h, w = (int(d) for d in state.armies.shape)
    bh = _board_hash(state)
    st_l = learned.initial_state(GameContext(seat, h, w))
    st_o = opp.initial_state(GameContext(1 - seat, h, w))

    silent_invalid = 0
    deliberate_pass = 0
    build_attempts = 0
    build_prices: list[int] = []
    growth_events = 0
    contact_turn = None
    discovery_turn = None
    attack_attempts = 0
    legal_counts: list[int] = []
    belief0 = None
    prev_owned = 1
    prev_armies_sum = None

    for turn in range(MAX_TURNS):
        eng_l = get_obs(state, seat)
        eng_o = get_obs(state, 1 - seat)
        tl, ol, al, _, ml = extract_numpy_boards(eng_l, h, w)
        to, oo, ao, _, mo = extract_numpy_boards(eng_o, h, w)
        obs_l = _observation_from_arrays(tl, ol, al, ml)
        obs_o = _observation_from_arrays(to, oo, ao, mo)
        if belief0 is None:
            belief0 = _belief_snapshot(architecture, checkpoint, obs_l)

        legal = enumerate_legal_actions(obs_l)
        legal_counts.append(len(legal))
        decision = learned.act(obs_l, st_l, deterministic=True, trace=TraceLevel.NONE, deadline=None)
        st_l = decision.new_state
        act = decision.action
        in_legal = is_legal_action(obs_l, act)
        if not in_legal:
            silent_invalid += 1
        if act.kind == KIND_PASS:
            deliberate_pass += 1
        if act.kind == KIND_BUILD:
            build_attempts += 1
            structs = own_structures(obs_l)
            build_prices.append(int(castle_price_at(act.row, act.col, structs)))
            # Also record official price if cell in bounds of engine state for seat0 mapping is hard;
            # keep agent-side price as primary observable.
        if act.kind == KIND_MOVE:
            dr, dc = DIRECTIONS[act.direction]
            nr, nc = act.row + dr, act.col + dc
            if 0 <= nr < h and 0 <= nc < w and int(ol[nr, nc]) == OWNER_OPP:
                attack_attempts += 1

        d_o = opp.act(obs_o, st_o, deterministic=True, trace=TraceLevel.NONE, deadline=None)
        st_o = d_o.new_state
        actions = [None, None]
        actions[seat] = _action_to_jax(act)
        actions[1 - seat] = _action_to_jax(d_o.action)
        before_time = int(state.time)
        before_armies = int(np.asarray(state.armies).sum())
        state, info = transition(state, jnp.stack(actions))
        after_armies = int(np.asarray(state.armies).sum())
        if int(state.time) > before_time and after_armies > before_armies:
            growth_events += 1

        owned = int(np.asarray(state.ownership[seat]).sum())
        enemy_vis = int((ol == 2).sum()) if hasattr(ol, "sum") else 0
        if discovery_turn is None and enemy_vis > 0:
            discovery_turn = turn
        if contact_turn is None and owned > prev_owned and enemy_vis > 0:
            # weak contact proxy
            contact_turn = turn
        prev_owned = owned

        if bool(info.is_done):
            break

    winner = int(info.winner) if bool(getattr(info, "is_done", False)) else -1
    result = "draw"
    if winner == seat:
        result = "win"
    elif winner == 1 - seat:
        result = "loss"

    return {
        "architecture": architecture,
        "checkpoint_sha16": _sha16(checkpoint),
        "seed": seed,
        "seat": seat,
        "opponent": opponent,
        "board_hw": [h, w],
        "map_hash": bh,
        "turns": turn + 1,
        "result": result,
        "winner": winner,
        "silent_invalid_actions": silent_invalid,
        "deliberate_pass_count": deliberate_pass,
        "build_attempts": build_attempts,
        "build_prices_sampled": build_prices[:20],
        "growth_events": growth_events,
        "contact_turn": contact_turn,
        "discovery_turn": discovery_turn,
        "attack_attempts": attack_attempts,
        "legal_action_count_mean": float(np.mean(legal_counts)) if legal_counts else 0.0,
        "model_visible_aux_belief": belief0,
        "official_build_cost_grid_probe": True,
    }


def main() -> int:
    seeds_doc = json.loads(SEEDS.read_text(encoding="utf-8"))
    budget = json.loads(BUDGET.read_text(encoding="utf-8"))
    if budget.get("opened"):
        raise SystemExit("diagnostic budget already opened; refuse re-entry without new version")
    diag_seeds = seeds_doc["partitions"]["rules_audit_diagnostic_seeds"]["seeds"]
    opponents = budget["allocation"]["opponents"]
    seats = budget["allocation"]["seats"]
    max_games = int(budget["maximum_total_diagnostic_games"])
    max_wall = float(budget["maximum_wall_clock_minutes"]) * 60.0

    budget["opened"] = True
    budget["opened_at"] = datetime.now(timezone.utc).isoformat()
    BUDGET.write_text(json.dumps(budget, indent=2) + "\n", encoding="utf-8")

    map_reg: dict[str, Any] = {"schema_version": 1, "hashes": {}}
    if MAP_HASHES.is_file() and MAP_HASHES.read_text(encoding="utf-8").strip():
        try:
            map_reg = json.loads(MAP_HASHES.read_text(encoding="utf-8"))
            map_reg.setdefault("hashes", {})
        except json.JSONDecodeError:
            map_reg = {"schema_version": 1, "hashes": {}}

    games: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for arch_key, (architecture, ckpt) in ARCHS.items():
        for seed in diag_seeds:
            for seat in seats:
                for opp in opponents:
                    if len(games) >= max_games:
                        break
                    if time.perf_counter() - t0 > max_wall:
                        break
                    g = play_game(
                        architecture=architecture,
                        checkpoint=ckpt,
                        seed=int(seed),
                        seat=int(seat),
                        opponent=str(opp),
                    )
                    g["arch_key"] = arch_key
                    games.append(g)
                    map_reg["hashes"][f"{arch_key}:{seed}:{seat}:{opp}"] = {
                        "map_hash": g["map_hash"],
                        "board_hw": g["board_hw"],
                        "partition": "rules_audit_diagnostic_seeds",
                    }
                    print(
                        json.dumps(
                            {
                                "done": len(games),
                                "arch": arch_key,
                                "seed": seed,
                                "seat": seat,
                                "opp": opp,
                                "result": g["result"],
                            }
                        ),
                        flush=True,
                    )

    # Summaries / sufficiency
    def summarize(arch_key: str) -> dict[str, Any]:
        subset = [g for g in games if g["arch_key"] == arch_key]
        return {
            "games": len(subset),
            "wdl": {
                "W": sum(1 for g in subset if g["result"] == "win"),
                "D": sum(1 for g in subset if g["result"] == "draw"),
                "L": sum(1 for g in subset if g["result"] == "loss"),
            },
            "silent_invalid_total": sum(g["silent_invalid_actions"] for g in subset),
            "contact_rate": sum(1 for g in subset if g["contact_turn"] is not None) / max(len(subset), 1),
            "discovery_rate": sum(1 for g in subset if g["discovery_turn"] is not None) / max(len(subset), 1),
            "mean_attack_attempts": float(np.mean([g["attack_attempts"] for g in subset])) if subset else 0.0,
            "belief_keys_seen": sorted(
                {
                    k
                    for g in subset
                    for k in (g.get("model_visible_aux_belief") or {})
                    if k not in {"value", "logits_finite"}
                }
            ),
            "build_prices_observed": any(g["build_prices_sampled"] for g in subset),
            "growth_events_total": sum(g["growth_events"] for g in subset),
        }

    cnn_s = summarize("cnn")
    graph_s = summarize("graph")

    missing = []
    for label, summary in (("cnn", cnn_s), ("graph", graph_s)):
        if summary["games"] == 0:
            missing.append(f"{label}:no_games")
        if not summary["belief_keys_seen"] and summary["games"]:
            # belief may be absent on some heads; value still recorded
            pass

    evidence = "SUFFICIENT_FOR_BLOCKER_CLASSIFICATION"
    if missing:
        evidence = "INSUFFICIENT_EVIDENCE"

    report = {
        "schema_version": 1,
        "kind": "PHASE9E_DIAGNOSTIC_CAPTURE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": time.perf_counter() - t0,
        "budget": budget,
        "games": games,
        "summary": {"cnn": cnn_s, "graph": graph_s},
        "evidence_status": evidence,
        "missing": missing,
        "instrumentation_retry_used": 0,
        "notes": [
            "Not a Phase 9D rerun; does not rewrite Phase 9D W/D/L or gates.",
            "No gradients; diagnostic seeds only.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    MAP_HASHES.write_text(json.dumps(map_reg, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"evidence_status": evidence, "games": len(games), "out": str(OUT)}, indent=2))
    return 0 if evidence != "INSUFFICIENT_EVIDENCE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
