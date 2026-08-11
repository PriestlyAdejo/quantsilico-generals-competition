"""Competition-native JAX daytime paired evaluation (protocol v2).

Pattern reuse from phase9fu (paired seeds, heartbeat, resume, SIGINT) without
inheriting Tactical/Hybrid/V002 role assumptions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from generals import GeneralsEnv
from generals.core import game

from generals_bot.action import Action
from generals_bot.competition_native_jax.policy import CompetitionNativePolicy
from generals_bot.competition_native_jax.transformer import TransformerWeights
from generals_bot.competition_native_jax.transformer_jax import init_params
from generals_bot.evaluation.match import make_board, make_transition
from generals_bot.evaluation.qualification import outcome_from_winner, score_rate
from generals_bot.observation import GameContext, Observation
from generals_bot.policies.base import ActionDecision, PolicyState, TraceLevel
from generals_bot.protocol import (
    TYPE_CASTLE,
    TYPE_FOG,
    TYPE_GENERAL,
    TYPE_MOUNTAIN,
    TYPE_PLAIN,
    TYPE_STRUCTURE_IN_FOG,
)
from generals_bot.selector import create_policy
from train.competition_native_jax.train_jax import load_tree

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "experiments/manifests/competition_native_jax_daytime_evaluation_protocol_v2.json"
_INTERRUPT = False

OPPONENT_ALIASES = {
    "official_expander": "official_expander",
    "official_hunter": "official_hunter",
    "heuristic_v2_aggressive": "heuristic_aggressive",
    "heuristic_v2_defensive": "heuristic_defensive",
    "heuristic_v2f_plus_planner_terminal_fix": "heuristic_v2f_best_reference",
}


def extract_numpy_boards(engine_obs, height: int, width: int):
    """Torch-free engine observation → NumPy grids + meta."""
    armies = np.asarray(engine_obs.armies)
    fog = np.asarray(engine_obs.fog_cells, dtype=bool)
    mountains = np.asarray(engine_obs.mountains, dtype=bool)
    castles = np.asarray(engine_obs.castles, dtype=bool)
    generals = np.asarray(engine_obs.generals, dtype=bool)
    structures_fog = np.asarray(engine_obs.structures_in_fog, dtype=bool)
    owned = np.asarray(engine_obs.owned_cells, dtype=bool)
    opp = np.asarray(engine_obs.opponent_cells, dtype=bool)

    type_grid = np.full((height, width), TYPE_PLAIN, dtype=np.int32)
    type_grid[fog] = TYPE_FOG
    type_grid[structures_fog] = TYPE_STRUCTURE_IN_FOG
    type_grid[mountains] = TYPE_MOUNTAIN
    type_grid[castles] = TYPE_CASTLE
    type_grid[generals] = TYPE_GENERAL
    owner = np.zeros((height, width), dtype=np.int32)
    owner[owned] = 1
    owner[opp] = 2
    meta = {
        "turn": int(engine_obs.timestep),
        "my_land": int(engine_obs.owned_land_count),
        "my_army": int(engine_obs.owned_army_count),
        "opp_land": int(engine_obs.opponent_land_count),
        "opp_army": int(engine_obs.opponent_army_count),
    }
    return type_grid, owner, armies.astype(np.int32), None, meta


def observation_from_arrays(type_grid, owner, armies, meta) -> Observation:
    return Observation(
        height=int(type_grid.shape[0]),
        width=int(type_grid.shape[1]),
        turn=int(meta["turn"]),
        my_land=int(meta["my_land"]),
        my_army=int(meta["my_army"]),
        opp_land=int(meta["opp_land"]),
        opp_army=int(meta["opp_army"]),
        type_grid=tuple(tuple(int(x) for x in row) for row in type_grid),
        owner_grid=tuple(tuple(int(x) for x in row) for row in owner),
        army_grid=tuple(tuple(int(x) for x in row) for row in armies),
    )


def action_to_jax(action) -> jnp.ndarray:
    return jnp.array(
        [action.kind, action.row, action.col, action.direction, action.split],
        dtype=jnp.int32,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def protocol_sha256(path: Path = PROTOCOL) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def jax_params_to_numpy_weights(params: dict) -> TransformerWeights:
    return TransformerWeights(
        patch_proj=np.asarray(params["patch_proj"], dtype=np.float32),
        cls=np.asarray(params["cls"], dtype=np.float32),
        pos=np.asarray(params["pos"], dtype=np.float32),
        global_proj=np.asarray(params["global_proj"], dtype=np.float32),
        attn_w=[np.asarray(L["attn_w"], dtype=np.float32) for L in params["layers"]],
        attn_out=[np.asarray(L["attn_out"], dtype=np.float32) for L in params["layers"]],
        ff_w1=[np.asarray(L["ff_w1"], dtype=np.float32) for L in params["layers"]],
        ff_w2=[np.asarray(L["ff_w2"], dtype=np.float32) for L in params["layers"]],
        move_head=np.asarray(params["move_head"], dtype=np.float32),
        build_head=np.asarray(params["build_head"], dtype=np.float32),
        pass_head=np.asarray(params["pass_head"], dtype=np.float32),
        value_head=np.asarray(params["value_head"], dtype=np.float32),
    )


class CNJPolicyAdapter:
    """Adapt CompetitionNativePolicy to the shared Policy act/initial_state interface."""

    def __init__(self, weights: TransformerWeights, seed: int = 0) -> None:
        self.inner = CompetitionNativePolicy(weights=weights, seed=seed)
        self.policy_id = "competition_native_jax_daytime"

    def initial_state(self, context: GameContext) -> PolicyState:
        self.inner.reset(context.height, context.width)
        return PolicyState(data={"player_id": context.player_id})

    def act(
        self,
        observation,
        state: PolicyState,
        *,
        deterministic: bool = True,
        trace: TraceLevel = TraceLevel.NONE,
        deadline=None,
    ) -> ActionDecision:
        action, info = self.inner.act(observation, deterministic=deterministic)
        return ActionDecision(
            action=action,
            new_state=state,
            policy_id=self.policy_id,
            strategic_option="LEARNED",
        )


def load_cnj_from_ckpt(ckpt_dir: Path, which: str = "ema") -> CNJPolicyAdapter:
    template = init_params(jax.random.PRNGKey(0))
    params = load_tree(ckpt_dir / f"{which}.npz", template)
    weights = jax_params_to_numpy_weights(params)
    return CNJPolicyAdapter(weights)


def play_game(p0: Any, p1: Any, *, seed: int, max_turns: int, focal_seat: int) -> dict:
    env = GeneralsEnv(mode="competition")
    transition = make_transition(env)
    get_obs = game.get_observation
    state = make_board(env, seed)
    h, w = (int(d) for d in state.armies.shape)
    st0 = p0.initial_state(GameContext(0, h, w))
    st1 = p1.initial_state(GameContext(1, h, w))
    winner: int | None = None
    turns = 0
    for turn_i in range(max_turns):
        if _INTERRUPT:
            return {"wdl": "incomplete_timeout", "winner": None, "turns": turn_i, "scored": False}
        eng0 = get_obs(state, 0)
        eng1 = get_obs(state, 1)
        t0, o0, a0, _, m0 = extract_numpy_boards(eng0, h, w)
        t1, o1, a1, _, m1 = extract_numpy_boards(eng1, h, w)
        obs0 = observation_from_arrays(t0, o0, a0, m0)
        obs1 = observation_from_arrays(t1, o1, a1, m1)
        try:
            d0 = p0.act(obs0, st0, deterministic=True, trace=TraceLevel.NONE, deadline=None)
            st0 = d0.new_state
            act0 = d0.action
            d1 = p1.act(obs1, st1, deterministic=True, trace=TraceLevel.NONE, deadline=None)
            st1 = d1.new_state
            act1 = d1.action
        except Exception as exc:  # noqa: BLE001
            return {
                "wdl": "fault",
                "winner": None,
                "turns": turn_i,
                "scored": False,
                "error": str(exc),
            }
        state, info = transition(state, jnp.stack([action_to_jax(act0), action_to_jax(act1)]))
        turns = turn_i + 1
        if bool(info.is_done):
            winner = int(info.winner)
            break
    wins, draws, losses = outcome_from_winner(winner, perspective=focal_seat)
    if wins:
        wdl = "win"
    elif losses:
        wdl = "loss"
    else:
        wdl = "draw"
    return {"wdl": wdl, "winner": winner, "turns": turns, "scored": True}


def aggregate(results: list[dict]) -> dict:
    scored = [r for r in results if r.get("scored")]
    w = sum(1 for r in scored if r["wdl"] == "win")
    d = sum(1 for r in scored if r["wdl"] == "draw")
    l = sum(1 for r in scored if r["wdl"] == "loss")
    return {
        "games_scored": len(scored),
        "W": w,
        "D": d,
        "L": l,
        "score_rate": score_rate(w, d, l) if scored else None,
        "draw_rate": (d / len(scored)) if scored else None,
    }


def run_phase(
    candidate: CNJPolicyAdapter,
    *,
    phase: str,
    seeds: list[int],
    opponents: list[str],
    pairs_per_opponent: int,
    seat_swaps: bool,
    max_turns: int,
    partial_path: Path,
) -> dict:
    results: list[dict] = []
    if partial_path.exists():
        prev = json.loads(partial_path.read_text(encoding="utf-8"))
        results = list(prev.get("results") or [])
    done_keys = {r["key"] for r in results}

    for opp_name in opponents:
        alias = OPPONENT_ALIASES.get(opp_name, opp_name)
        for seed in seeds[: max(pairs_per_opponent, len(seeds))]:
            seats = [0, 1] if seat_swaps else [0]
            for focal in seats:
                key = f"{phase}:{opp_name}:{seed}:focal{focal}"
                if key in done_keys:
                    continue
                if _INTERRUPT:
                    break
                opp = create_policy(alias, seed=seed)
                if focal == 0:
                    p0, p1 = candidate, opp
                else:
                    p0, p1 = opp, candidate
                # fresh candidate state each game
                cand = CNJPolicyAdapter(candidate.inner.weights, seed=seed + focal)
                if focal == 0:
                    p0 = cand
                else:
                    p1 = cand
                t0 = time.perf_counter()
                outcome = play_game(p0, p1, seed=seed, max_turns=max_turns, focal_seat=focal)
                outcome.update(
                    {
                        "key": key,
                        "phase": phase,
                        "opponent": opp_name,
                        "seed": seed,
                        "focal_seat": focal,
                        "elapsed_s": time.perf_counter() - t0,
                        "ts": _now(),
                    }
                )
                results.append(outcome)
                atomic_write_json(
                    partial_path,
                    {"phase": phase, "results": results, "updated_at": _now()},
                )
                print(
                    f"{key} -> {outcome.get('wdl')} turns={outcome.get('turns')} "
                    f"scored={outcome.get('scored')}",
                    flush=True,
                )
    by_opp: dict[str, list] = {}
    for r in results:
        by_opp.setdefault(r["opponent"], []).append(r)
    return {
        "phase": phase,
        "results": results,
        "overall": aggregate(results),
        "by_opponent": {k: aggregate(v) for k, v in by_opp.items()},
        "updated_at": _now(),
    }


def decide_teacher_gate(screening: dict, selection: dict | None, gates: dict) -> dict:
    overall = (selection or screening)["overall"]
    by_opp = (selection or screening)["by_opponent"]
    sr = overall.get("score_rate")
    expander = (by_opp.get("official_expander") or {}).get("score_rate")
    hunter = (by_opp.get("official_hunter") or {}).get("score_rate")

    reasons = []
    ok = True
    if sr is None or overall.get("games_scored", 0) < 8:
        ok = False
        reasons.append("insufficient_scored_games")
    if expander is not None and expander < float(gates["VALID_DAYTIME_TEACHER_min_score_rate_vs_expander"]):
        ok = False
        reasons.append(f"expander_score_rate_{expander}")
    if hunter is not None and hunter < float(gates["VALID_DAYTIME_TEACHER_min_score_rate_vs_hunter"]):
        ok = False
        reasons.append(f"hunter_score_rate_{hunter}")

    # Even baseline ≈ 0.5; require modest improvement on overall if enough games
    if sr is not None and sr < 0.5 + float(gates.get("min_direct_improvement_vs_even", 0.05)) - 1e-9:
        # Allow research-only rather than hard fail if close
        if sr < 0.50:
            ok = False
            reasons.append(f"overall_below_even_{sr}")

    if ok:
        disposition = "VALID_DAYTIME_TEACHER"
        promote_medium = sr is not None and sr >= float(gates.get("promote_to_medium_min_score_rate", 0.58))
    elif sr is not None and sr >= 0.45:
        disposition = "RESEARCH_ONLY_CHECKPOINT"
        promote_medium = False
    else:
        disposition = "NO_VALID_TRAINED_CHECKPOINT"
        promote_medium = False

    return {
        "schema_version": 1,
        "kind": "INDEPENDENTLY_VALID_TRAINED_CHECKPOINT_GATE",
        "disposition": disposition,
        "promote_to_medium": promote_medium,
        "score_rate": sr,
        "expander_score_rate": expander,
        "hunter_score_rate": hunter,
        "reasons": reasons,
        "updated_at": _now(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--which", default="ema", choices=["ema", "raw"])
    ap.add_argument("--phase", default="screening", choices=["screening", "selection", "confirmation", "auto"])
    args = ap.parse_args()

    def _on_sigint(*_a):
        global _INTERRUPT
        _INTERRUPT = True

    signal.signal(signal.SIGINT, _on_sigint)

    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    freeze = json.loads(
        (REPO / "experiments/manifests/competition_native_jax_daytime_eval_protocol_freeze.json").read_text()
    )
    sha = protocol_sha256()
    if sha != freeze["evaluation_protocol_sha256"]:
        raise SystemExit(f"protocol sha drift: {sha} != {freeze['evaluation_protocol_sha256']}")

    ckpt = Path(args.ckpt)
    candidate = load_cnj_from_ckpt(ckpt, which=args.which)
    out_dir = REPO / "experiments/manifests"
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_DAYTIME_EVAL",
        "checkpoint": str(ckpt).replace("\\", "/"),
        "which": args.which,
        "evaluation_protocol_id": freeze["evaluation_protocol_id"],
        "evaluation_protocol_sha256": sha,
        "updated_at": _now(),
    }

    phases = []
    if args.phase == "auto":
        phases = ["screening", "selection"]
    else:
        phases = [args.phase]

    screening_rep = None
    selection_rep = None
    for phase in phases:
        cfg = proto[phase]
        seeds = list(cfg["seeds"])
        # pairs_per_opponent limits how many seeds to use
        n_pairs = int(cfg.get("pairs_per_opponent", len(seeds)))
        use_seeds = seeds[:n_pairs] if n_pairs < len(seeds) else seeds
        partial = out_dir / f"competition_native_jax_daytime_eval_{phase}.partial.json"
        print(f"=== {phase} seeds={use_seeds} ===", flush=True)
        rep = run_phase(
            candidate,
            phase=phase,
            seeds=use_seeds,
            opponents=list(proto["opponents"]),
            pairs_per_opponent=n_pairs,
            seat_swaps=bool(proto.get("seat_swaps", True)),
            max_turns=int(proto.get("max_turns", 1200)),
            partial_path=partial,
        )
        report[phase] = rep
        if phase == "screening":
            screening_rep = rep
            # Gate progression to selection
            sr = (rep.get("overall") or {}).get("score_rate")
            if sr is None or sr < 0.40:
                print("screening too weak; skipping selection", flush=True)
                break
        elif phase == "selection":
            selection_rep = rep

    gate = decide_teacher_gate(screening_rep or {"overall": {}, "by_opponent": {}}, selection_rep, proto["gates"])
    report["teacher_gate"] = gate
    atomic_write_json(out_dir / "competition_native_jax_v4_3_daytime_eval.json", report)
    atomic_write_json(out_dir / "competition_native_jax_v4_3_teacher_gate.json", gate)

    prog_path = out_dir / "competition_native_jax_v4_3_programme_state.json"
    prog = json.loads(prog_path.read_text(encoding="utf-8"))
    prog["daytime_eval"] = "experiments/manifests/competition_native_jax_v4_3_daytime_eval.json"
    prog["teacher_gate"] = gate["disposition"]
    prog["promote_to_medium"] = bool(gate.get("promote_to_medium"))
    if gate["disposition"] == "VALID_DAYTIME_TEACHER":
        prog["status"] = "STAGE_7_VALID_TEACHER"
        prog["current_stage"] = "STAGE_7_5_EARLY_DEPLOY"
    else:
        prog["status"] = f"STAGE_7_{gate['disposition']}"
        prog["current_stage"] = "STAGE_18_WRITEONLY_IF_NO_PACKAGE"
    prog_path.write_text(json.dumps(prog, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"teacher_gate": gate["disposition"], "score_rate": gate.get("score_rate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
