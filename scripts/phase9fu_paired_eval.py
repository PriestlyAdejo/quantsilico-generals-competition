"""Phase 9FU Stage 3 paired evaluation under frozen protocol (resumable).

Implementation-only amendment: progress, checkpoints, resume, CLI filters,
heartbeat, wall-time monitoring, SIGINT flush. Protocol v1 seeds/maps/seats/
opponents/scoring/thresholds are unchanged.

TIMEOUT_PROTOCOL_RULE: wall timeout classifies INCOMPLETE_TIMEOUT and does not
alter score_rate / WDL of completed pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import jax.numpy as jnp
from generals import GeneralsEnv
from generals.core import game
from generals_bot.evaluation.match import make_board, make_transition
from generals_bot.evaluation.qualification import outcome_from_winner, score_rate
from generals_bot.observation import GameContext
from generals_bot.policies.base import TraceLevel
from generals_bot.selector import create_policy
from generals_bot.training.bridge_benchmark import extract_numpy_boards
from generals_bot.training.collect_bc import _action_to_jax, _observation_from_arrays

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "experiments" / "manifests" / "phase9fu_evaluation_protocol.json"
MANIFESTS = REPO / "experiments" / "manifests"
EVALUATION_RUNTIME = "source_tree_policy_factory"

_INTERRUPT = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def protocol_sha256(path: Path = PROTOCOL) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partial_path(candidate_id: str) -> Path:
    safe = candidate_id.replace("/", "_").replace("\\", "_")
    return MANIFESTS / f"phase9fu_eval_{safe}.partial.json"


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


def load_checkpoint(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pair_key(opponent: str, seed: int) -> str:
    return f"{opponent}:{seed}"


def rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        try:
            import resource

            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        except Exception:
            return None


def _play_instance_game(
    p0: Any,
    p1: Any,
    *,
    seed: int,
    max_turns: int,
    focal_seat: int,
    game_wall_s: float | None = None,
) -> dict:
    env = GeneralsEnv(mode="competition")
    transition = make_transition(env)
    get_obs = game.get_observation
    state = make_board(env, seed)
    h, w = (int(d) for d in state.armies.shape)
    st0 = p0.initial_state(GameContext(0, h, w))
    st1 = p1.initial_state(GameContext(1, h, w))
    winner: int | None = None
    turns = 0
    timed_out = False
    t_game0 = time.perf_counter()
    for turn_i in range(max_turns):
        if _INTERRUPT:
            timed_out = True
            break
        if game_wall_s is not None and (time.perf_counter() - t_game0) > game_wall_s:
            timed_out = True
            break
        eng0 = get_obs(state, 0)
        eng1 = get_obs(state, 1)
        t0, o0, a0, _, m0 = extract_numpy_boards(eng0, h, w)
        t1, o1, a1, _, m1 = extract_numpy_boards(eng1, h, w)
        obs0 = _observation_from_arrays(t0, o0, a0, m0)
        obs1 = _observation_from_arrays(t1, o1, a1, m1)
        try:
            d0 = p0.act(obs0, st0, deterministic=True, trace=TraceLevel.NONE, deadline=None)
            st0 = d0.new_state
            act0 = d0.action
        except Exception as exc:
            return _protocol_fault_result(
                seat=0,
                turn=turn_i,
                seed=seed,
                exc=exc,
                focal_seat=focal_seat,
                turns=turn_i,
            )
        try:
            d1 = p1.act(obs1, st1, deterministic=True, trace=TraceLevel.NONE, deadline=None)
            st1 = d1.new_state
            act1 = d1.action
        except Exception as exc:
            return _protocol_fault_result(
                seat=1,
                turn=turn_i,
                seed=seed,
                exc=exc,
                focal_seat=focal_seat,
                turns=turn_i,
            )
        state, info = transition(state, jnp.stack([_action_to_jax(act0), _action_to_jax(act1)]))
        turns = turn_i + 1
        if bool(info.is_done):
            winner = int(info.winner)
            break
    if timed_out and winner is None:
        return {
            "wdl": "incomplete_timeout",
            "winner": None,
            "turns": turns,
            "timeout_class": "INCOMPLETE_TIMEOUT",
            "scored": False,
        }
    wins, draws, losses = outcome_from_winner(winner, perspective=focal_seat)
    if wins:
        wdl = "win"
    elif losses:
        wdl = "loss"
    else:
        wdl = "draw"
    return {"wdl": wdl, "winner": winner, "turns": turns, "scored": True}


def _protocol_fault_result(
    *,
    seat: int,
    turn: int,
    seed: int,
    exc: BaseException,
    focal_seat: int,
    turns: int,
) -> dict:
    """Record a policy exception as PROTOCOL_FAULT — never silently PASS."""
    import hashlib
    import traceback

    tb = traceback.format_exc()
    tb_hash = hashlib.sha256(tb.encode("utf-8")).hexdigest()
    return {
        "wdl": "protocol_fault",
        "winner": None,
        "turns": turns,
        "scored": False,
        "fault_class": "PROTOCOL_FAULT",
        "fault": {
            "seat": seat,
            "turn": turn,
            "seed": seed,
            "focal_seat": focal_seat,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc)[:500],
            "traceback_sha256": tb_hash,
        },
    }


def _pair(
    make_challenger: Callable[[], Any],
    opponent_name: str,
    *,
    seed: int,
    max_turns: int = 1200,
    game_wall_s: float | None = None,
) -> dict:
    games = []
    for swap in (False, True):
        if _INTERRUPT:
            break
        ch = make_challenger()
        opp = create_policy(opponent_name, seed=seed + 1)
        if not swap:
            p0, p1, focal = ch, opp, 0
        else:
            p0, p1, focal = opp, ch, 1
        t0 = time.perf_counter()
        g = _play_instance_game(
            p0,
            p1,
            seed=seed,
            max_turns=max_turns,
            focal_seat=focal,
            game_wall_s=game_wall_s,
        )
        g["elapsed_s"] = time.perf_counter() - t0
        g["swap"] = swap
        g["seed"] = seed
        g["seat"] = focal
        games.append(g)
    return {"seed": seed, "opponent": opponent_name, "games": games}


def _summarize(pairs: list[dict]) -> dict:
    wins = draws = losses = 0
    unscored = 0
    protocol_faults = 0
    for p in pairs:
        for g in p["games"]:
            if g.get("fault_class") == "PROTOCOL_FAULT" or g.get("wdl") == "protocol_fault":
                protocol_faults += 1
                unscored += 1
                continue
            if g.get("scored") is False or g.get("wdl") == "incomplete_timeout":
                unscored += 1
                continue
            if g["wdl"] == "win":
                wins += 1
            elif g["wdl"] == "draw":
                draws += 1
            else:
                losses += 1
    n = wins + draws + losses
    return {
        "games": n,
        "pairs": len(pairs),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "unscored_incomplete_timeout_games": unscored,
        "protocol_fault_games": protocol_faults,
        "score_rate": score_rate(wins, draws, losses) if n else 0.0,
        "draw_rate": draws / max(1, n),
    }


def planned_pairs(
    proto: dict,
    *,
    opponents: list[str] | None = None,
) -> list[tuple[str, int]]:
    seeds = list(proto["seeds"])
    n_direct = int(proto["counts"]["direct_vs_v001_paired"])
    n_base = int(proto["counts"]["per_critical_baseline_paired"])
    v001 = "heuristic_v2f_plus_planner_terminal_fix"
    baselines = list(proto["critical_baselines"])
    # Keep pair set identical to the original Stage 3 runner (critical baselines only).
    # Castle inclusion would change evaluation counts and requires protocol v2.
    out: list[tuple[str, int]] = []
    if opponents is None or v001 in opponents:
        for s in seeds[:n_direct]:
            out.append((v001, s))
    for base in baselines:
        if opponents is not None and base not in opponents:
            continue
        for s in seeds[:n_base]:
            out.append((base, s + 1000))
    return out


def empty_partial(
    *,
    candidate_id: str,
    proto: dict,
    proto_hash: str,
    planned: list[tuple[str, int]],
) -> dict:
    return {
        "schema_version": 1,
        "kind": "PHASE9FU_EVAL_PARTIAL",
        "candidate_id": candidate_id,
        "protocol_version": proto["protocol_version"],
        "protocol_sha256": proto_hash,
        "evaluation_runtime": EVALUATION_RUNTIME,
        "created_at": _now(),
        "updated_at": _now(),
        "status": "IN_PROGRESS",
        "planned_pairs": [{"opponent": o, "seed": s} for o, s in planned],
        "completed_pairs": {},
        "completed_pair_order": [],
        "interrupted": False,
        "incomplete_timeout": False,
    }


def validate_resume(partial: dict, *, candidate_id: str, proto_hash: str, planned: list[tuple[str, int]]) -> None:
    if partial.get("candidate_id") != candidate_id:
        raise SystemExit(
            f"resume candidate mismatch: {partial.get('candidate_id')} != {candidate_id}"
        )
    if partial.get("protocol_sha256") != proto_hash:
        raise SystemExit("resume protocol hash mismatch")
    saved = {(p["opponent"], p["seed"]) for p in partial.get("planned_pairs") or []}
    want = set(planned)
    if saved and saved != want:
        raise SystemExit("resume planned pair set mismatch")


def classify_candidate(cand: dict, thr: dict, n_direct: int) -> dict:
    direct = cand["direct_vs_v001"]["summary"]
    pairs_ok = direct["pairs"] >= n_direct
    improvement = direct["score_rate"] - 0.5
    suite_floor = 0.5 - float(thr["max_opponent_suite_score_rate_regression"])
    suite_ok = cand["suite_score_rate_mean"] >= suite_floor
    draw_ok = direct["draw_rate"] <= 0.5 + float(thr["max_draw_rate_increase"])
    if cand.get("status") == "INCOMPLETE_TIMEOUT":
        label = "ABORTED_INCOMPLETE"
    elif not pairs_ok:
        label = "PROMISING_BUT_UNDER_SAMPLED"
    elif (
        improvement >= float(thr["min_direct_score_rate_improvement_vs_v001"])
        and suite_ok
        and draw_ok
    ):
        label = "V002_ELIGIBLE"
    elif improvement > 0 and suite_ok:
        label = "PROMISING_BUT_UNDER_SAMPLED"
    else:
        label = "REJECTED"
    return {
        "label": label,
        "direct_improvement_vs_even": improvement,
        "suite_score_rate_mean": cand["suite_score_rate_mean"],
        "pairs_ok": pairs_ok,
        "suite_ok": suite_ok,
        "draw_ok": draw_ok,
    }


def build_factories(eligible: list[str]) -> dict[str, Callable[[], Any]]:
    factories: dict[str, Callable[[], Any]] = {}
    if "QS-P9FU-HEURISTIC-TACTICAL-V2" in eligible:
        factories["QS-P9FU-HEURISTIC-TACTICAL-V2"] = lambda: create_policy(
            "heuristic_v2f_tactical_attack_v2", seed=0
        )
    if "QS-P9FU-HYBRID-BC-V1" in eligible:
        from generals_bot.policies.hybrid_bc_ranker import HybridBcRankerPolicy

        ckpt = REPO / "experiments/phase9f_cnn_ranker_v1/checkpoints/bc/model.json"
        factories["QS-P9FU-HYBRID-BC-V1"] = lambda: HybridBcRankerPolicy(
            checkpoint_json=ckpt, device="cpu"
        )
    return factories


def aggregate_from_partial(partial: dict, proto: dict) -> dict:
    v001 = "heuristic_v2f_plus_planner_terminal_fix"
    baselines = list(proto["critical_baselines"])
    pairs = list(partial.get("completed_pairs", {}).values())
    direct = [p for p in pairs if p["opponent"] == v001]
    suite = {}
    for base in baselines:
        bp = [p for p in pairs if p["opponent"] == base]
        suite[base] = {"summary": _summarize(bp), "pairs_n": len(bp)}
    suite_mean = (
        sum(suite[b]["summary"]["score_rate"] for b in baselines) / max(1, len(baselines))
        if baselines
        else 0.0
    )
    status = partial.get("status", "COMPLETE")
    return {
        "direct_vs_v001": {"summary": _summarize(direct), "pairs_n": len(direct)},
        "external_suite": suite,
        "suite_score_rate_mean": suite_mean,
        "status": status,
        "evaluation_runtime": EVALUATION_RUNTIME,
        "completed_pairs_n": len(pairs),
    }


class Heartbeat:
    def __init__(self, interval_s: float = 60.0) -> None:
        self.interval_s = interval_s
        self.state: dict[str, Any] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.emissions = 0

    def update(self, **kwargs: Any) -> None:
        self.state.update(kwargs)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="eval-heartbeat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            payload = {
                "event": "heartbeat",
                "ts": _now(),
                "rss_bytes": rss_bytes(),
                **self.state,
            }
            print(json.dumps(payload), flush=True)
            self.emissions += 1


def evaluate_candidate(
    *,
    candidate_id: str,
    factory: Callable[[], Any],
    proto: dict,
    proto_hash: str,
    planned: list[tuple[str, int]],
    resume_from: Path | None,
    game_wall_s: float | None,
    candidate_wall_s: float | None,
    heartbeat_s: float,
) -> dict:
    global _INTERRUPT
    ckpt_path = partial_path(candidate_id)
    if resume_from is not None:
        partial = load_checkpoint(resume_from)
        validate_resume(partial, candidate_id=candidate_id, proto_hash=proto_hash, planned=planned)
    elif ckpt_path.exists():
        partial = load_checkpoint(ckpt_path)
        validate_resume(partial, candidate_id=candidate_id, proto_hash=proto_hash, planned=planned)
    else:
        partial = empty_partial(
            candidate_id=candidate_id, proto=proto, proto_hash=proto_hash, planned=planned
        )

    done = set(partial.get("completed_pair_order") or [])
    hb = Heartbeat(interval_s=heartbeat_s)
    t_cand0 = time.perf_counter()
    hb.update(
        candidate=candidate_id,
        phase="starting",
        completed_pairs=len(done),
        total_expected_pairs=len(planned),
        completed_games=sum(
            len(partial["completed_pairs"][k]["games"]) for k in done if k in partial["completed_pairs"]
        ),
        elapsed_s=0.0,
    )
    hb.start()

    def _flush(status: str) -> None:
        partial["status"] = status
        partial["updated_at"] = _now()
        partial["interrupted"] = status == "INTERRUPTED"
        partial["incomplete_timeout"] = status == "INCOMPLETE_TIMEOUT"
        atomic_write_json(ckpt_path, partial)

    try:
        for idx, (opponent, seed) in enumerate(planned, start=1):
            key = pair_key(opponent, seed)
            if key in done:
                continue
            if _INTERRUPT:
                _flush("INTERRUPTED")
                break
            if candidate_wall_s is not None and (time.perf_counter() - t_cand0) > candidate_wall_s:
                _flush("INCOMPLETE_TIMEOUT")
                break
            hb.update(
                candidate=candidate_id,
                opponent=opponent,
                current_seed=seed,
                phase="playing_pair",
                pair_index=idx,
                total_expected_pairs=len(planned),
                completed_pairs=len(done),
                elapsed_s=time.perf_counter() - t_cand0,
            )
            print(
                json.dumps(
                    {
                        "event": "pair_start",
                        "candidate": candidate_id,
                        "opponent": opponent,
                        "seed": seed,
                        "pair_index": idx,
                        "total_pairs": len(planned),
                    }
                ),
                flush=True,
            )
            result = _pair(
                factory,
                opponent,
                seed=seed,
                game_wall_s=game_wall_s,
            )
            if len(result["games"]) < 2:
                _flush("INTERRUPTED")
                break
            for g in result["games"]:
                print(
                    json.dumps(
                        {
                            "event": "game_complete",
                            "candidate": candidate_id,
                            "opponent": opponent,
                            "seed": seed,
                            "pair_index": idx,
                            "total_pairs": len(planned),
                            "seat": g.get("seat"),
                            "swap": g.get("swap"),
                            "result": g.get("wdl"),
                            "turns": g.get("turns"),
                            "elapsed_game_s": g.get("elapsed_s"),
                            "elapsed_candidate_s": time.perf_counter() - t_cand0,
                            "completed_pairs": len(done),
                            "scored": g.get("scored", True),
                        }
                    ),
                    flush=True,
                )
            # Only commit fully completed pairs (both seats).
            if any(g.get("scored") is False for g in result["games"]):
                # Timed-out mid-pair: do not commit; classify incomplete.
                _flush("INCOMPLETE_TIMEOUT")
                break
            partial["completed_pairs"][key] = result
            partial["completed_pair_order"].append(key)
            done.add(key)
            _flush("IN_PROGRESS")
            print(
                json.dumps(
                    {
                        "event": "pair_complete",
                        "candidate": candidate_id,
                        "opponent": opponent,
                        "seed": seed,
                        "pair_index": idx,
                        "total_pairs": len(planned),
                        "completed_pairs": len(done),
                        "elapsed_candidate_s": time.perf_counter() - t_cand0,
                    }
                ),
                flush=True,
            )
        else:
            _flush("COMPLETE")
    finally:
        hb.stop()

    agg = aggregate_from_partial(partial, proto)
    completed_path = MANIFESTS / f"phase9fu_eval_{candidate_id.replace('/', '_')}.complete.json"
    atomic_write_json(
        completed_path,
        {
            "schema_version": 1,
            "kind": "PHASE9FU_EVAL_COMPLETE",
            "created_at": _now(),
            "candidate_id": candidate_id,
            "protocol_version": proto["protocol_version"],
            "protocol_sha256": proto_hash,
            "evaluation_runtime": EVALUATION_RUNTIME,
            "partial_status": partial.get("status"),
            "aggregate": agg,
            "partial_path": str(ckpt_path.relative_to(REPO).as_posix()),
        },
    )
    return {"partial": partial, "aggregate": agg, "checkpoint": ckpt_path}


def write_final_manifest(
    *,
    results: dict,
    classifications: dict,
    recommendation: str | None,
    proto: dict,
    proto_hash: str,
    diagnostic: dict | None = None,
    aborted: dict | None = None,
) -> Path:
    out = {
        "schema_version": 1,
        "kind": "PHASE9FU_PAIRED_EVAL",
        "created_at": _now(),
        "protocol_version": proto["protocol_version"],
        "protocol_sha256": proto_hash,
        "evaluation_runtime": EVALUATION_RUNTIME,
        "results": results,
        "classifications": classifications,
        "recommendation": recommendation,
        "upload_this_status": (
            "RECOMMENDED" if recommendation else "NO_CANDIDATE_CURRENTLY_RECOMMENDED"
        ),
        "diagnostic_only": diagnostic or {},
        "aborted_incomplete": aborted or {},
    }
    path = MANIFESTS / "phase9fu_v001_vs_challengers.json"
    atomic_write_json(path, out)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 9FU resumable paired evaluator")
    p.add_argument("--candidate", action="append", dest="candidates", default=None)
    p.add_argument("--opponent", action="append", dest="opponents", default=None)
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument("--max-game-wall-s", type=float, default=None)
    p.add_argument("--max-candidate-wall-s", type=float, default=None)
    p.add_argument("--heartbeat-s", type=float, default=60.0)
    p.add_argument("--write-recommendation", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global _INTERRUPT
    args = parse_args(argv)
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    proto_hash = protocol_sha256()
    behav = json.loads(
        (REPO / "experiments" / "manifests" / "phase9fu_behavioural_gates.json").read_text(
            encoding="utf-8"
        )
    )
    eligible = list(behav.get("stage3_eligible") or [])
    factories = build_factories(eligible)
    selected = list(args.candidates) if args.candidates else list(factories.keys())
    for cid in selected:
        if cid not in factories:
            raise SystemExit(f"unknown or ineligible candidate: {cid}")

    def _handle_sigint(_signum: int, _frame: Any) -> None:
        global _INTERRUPT
        _INTERRUPT = True
        print(json.dumps({"event": "signal", "status": "INTERRUPTED_REQUESTED"}), flush=True)

    signal.signal(signal.SIGINT, _handle_sigint)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_sigint)

    planned = planned_pairs(proto, opponents=args.opponents)
    thr = proto["thresholds"]
    n_direct = int(proto["counts"]["direct_vs_v001_paired"])
    results: dict = {"candidates": {}}
    classifications: dict = {}

    for cid in selected:
        print(f"evaluating {cid} ...", flush=True)
        out = evaluate_candidate(
            candidate_id=cid,
            factory=factories[cid],
            proto=proto,
            proto_hash=proto_hash,
            planned=planned,
            resume_from=args.resume_from if len(selected) == 1 else None,
            game_wall_s=args.max_game_wall_s,
            candidate_wall_s=args.max_candidate_wall_s,
            heartbeat_s=args.heartbeat_s,
        )
        results["candidates"][cid] = out["aggregate"]
        classifications[cid] = classify_candidate(out["aggregate"], thr, n_direct)
        print(
            json.dumps(
                {
                    "cid": cid,
                    "direct": out["aggregate"]["direct_vs_v001"]["summary"],
                    "suite_mean": out["aggregate"]["suite_score_rate_mean"],
                    "status": out["aggregate"].get("status"),
                    "classification": classifications[cid],
                }
            ),
            flush=True,
        )

    eligible_v002 = [c for c, v in classifications.items() if v["label"] == "V002_ELIGIBLE"]
    recommendation = None
    if len(eligible_v002) == 1:
        recommendation = eligible_v002[0]
    elif len(eligible_v002) > 1:
        ordered = sorted(
            eligible_v002,
            key=lambda c: results["candidates"][c]["direct_vs_v001"]["summary"]["score_rate"],
            reverse=True,
        )
        tol = float(thr["tie_tolerance"]["score_rate"])
        s0 = results["candidates"][ordered[0]]["direct_vs_v001"]["summary"]["score_rate"]
        s1 = results["candidates"][ordered[1]]["direct_vs_v001"]["summary"]["score_rate"]
        if abs(s0 - s1) > tol:
            recommendation = ordered[0]
        else:
            for c in ordered:
                classifications[c]["label"] = "PROMISING_BUT_UNDER_SAMPLED"
                classifications[c]["tie_break"] = "indistinguishable_direct_score"

    write_final_manifest(
        results=results,
        classifications=classifications,
        recommendation=recommendation if args.write_recommendation else None,
        proto=proto,
        proto_hash=proto_hash,
    )

    if args.write_recommendation:
        if recommendation:
            pkg_root = REPO / "submission" / "packages" / recommendation
            builds = sorted(pkg_root.glob("*/package.zip")) if pkg_root.exists() else []
            if builds:
                z = builds[-1]
                sha_path = z.parent / "sha256.txt"
                sha = sha_path.read_text(encoding="utf-8").strip() if sha_path.exists() else ""
                rec = {
                    "status": "RECOMMENDED_FOR_MANUAL_UPLOAD_WITH_EXTERNAL_LINUX_BLOCKER",
                    "candidate_id": recommendation,
                    "build_hash": z.parent.name,
                    "package_path": str(z.relative_to(REPO).as_posix()),
                    "sha256": sha,
                    "public_version_proposed": "QS-PUBLIC-V002",
                }
                (REPO / "submission" / "roles" / "recommended.json").write_text(
                    json.dumps(rec, indent=2) + "\n", encoding="utf-8"
                )
                (REPO / "submission" / "UPLOAD_THIS.md").write_text(
                    f"Current recommendation:\n{recommendation}\n\n"
                    f"Package:\n{rec['package_path']}\n\n"
                    f"SHA-256:\n{sha}\n\n"
                    f"Status:\n{rec['status']}\n\n"
                    "Do not upload automatically.\n",
                    encoding="utf-8",
                )
        else:
            (REPO / "submission" / "roles" / "recommended.json").write_text(
                json.dumps(
                    {
                        "status": "NO_CANDIDATE_CURRENTLY_RECOMMENDED",
                        "package_path": None,
                        "reason": "No challenger passed V002 gates under frozen protocol v1",
                        "classifications": classifications,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    print(json.dumps({"recommendation": recommendation, "classifications": classifications}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
