"""STAGE5_TEACHER_R1 STEP1: generate HunterAgent teacher games with transcripts.

Mirrors generals_bot.evaluation.match.run_python_agent_match (pinned engine
authority) but records both players' decoded actions per turn, then verifies
each transcript by deterministic replay through the same engine transition
(no agents): the replayed winner/turn count must match the live match.

Predeclared gate (stage5_teacher_r1_plan.yaml): hunter must WIN >= 16/20
games with zero engine faults before any labels are extracted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import jax.numpy as jnp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from generals import GeneralsEnv  # noqa: E402
from generals_bot.evaluation.match import (  # noqa: E402
    _load_engine_protocol,
    make_board,
    make_transition,
)
from generals.core import game  # noqa: E402

HUNTER = REPO / "baselines/official_hunter/main.py"
OPPONENT = REPO / "baselines/heuristic_v0/main.py"
OUT_ROOT = REPO / "experiments/marathon/teacher_r1/step1_generation"


def play_recorded(seed: int, max_turns: int = 1200) -> dict:
    protocol = _load_engine_protocol()
    decode_action = protocol.decode_action
    encode_handshake = protocol.encode_handshake
    encode_observation = protocol.encode_observation

    env = GeneralsEnv(mode="competition")
    get_obs = game.get_full_observation if env.perfect_info else game.get_observation
    state = make_board(env, seed)
    height, width = (int(d) for d in state.armies.shape)
    transition = make_transition(env)
    truncation = min(env.truncation, max_turns)

    def spawn(main_path: Path, player_id: int) -> subprocess.Popen[str]:
        import os

        env_vars = os.environ.copy()
        src = str(REPO / "src")
        env_vars["PYTHONPATH"] = src + os.pathsep + env_vars.get("PYTHONPATH", "")
        proc = subprocess.Popen(
            [sys.executable, "-u", str(main_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(main_path.parent),
            env=env_vars,
        )
        assert proc.stdin is not None
        proc.stdin.write(encode_handshake(player_id, height, width))
        proc.stdin.flush()
        return proc

    p0 = spawn(HUNTER, 0)
    p1 = spawn(OPPONENT, 1)
    faults = [0, 0]
    crashed = [False, False]
    stderr_tails = ["", ""]
    actions0: list[list[int]] = []
    actions1: list[list[int]] = []
    t0 = time.perf_counter()
    turn = 0
    winner = -1
    truncated = False

    def _stderr_tail(proc: subprocess.Popen[str]) -> str:
        try:
            if proc.stderr is None:
                return ""
            return proc.stderr.read()[-2000:]
        except Exception:
            return ""

    def ask(proc: subprocess.Popen[str], obs, idx: int) -> jnp.ndarray:
        assert proc.stdin is not None and proc.stdout is not None
        try:
            proc.stdin.write(encode_observation(obs))
            proc.stdin.flush()
        except OSError as exc:
            crashed[idx] = True
            stderr_tails[idx] = f"{exc!r} | exit={proc.poll()} | {_stderr_tail(proc)}"
            raise
        line = proc.stdout.readline()
        if not line:
            faults[idx] += 1
            return jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
        try:
            return decode_action(line)
        except Exception:
            faults[idx] += 1
            return jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)

    try:
        while turn < truncation:
            obs0 = get_obs(state, 0)
            obs1 = get_obs(state, 1)
            try:
                a0 = ask(p0, obs0, 0)
                a1 = ask(p1, obs1, 1)
            except OSError:
                winner = 1 if crashed[0] else 0
                break
            actions0.append([int(x) for x in a0])
            actions1.append([int(x) for x in a1])
            state, info = transition(state, jnp.stack([a0, a1]))
            turn += 1
            if bool(info.is_done):
                winner = int(info.winner)
                break
        else:
            truncated = True
    finally:
        for proc in (p0, p1):
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    return {
        "seed": seed,
        "winner": winner,
        "turns": turn,
        "truncated": truncated,
        "faults_hunter": faults[0],
        "faults_opponent": faults[1],
        "crashed_hunter": crashed[0],
        "crashed_opponent": crashed[1],
        "elapsed_s": time.perf_counter() - t0,
        "actions_hunter": actions0,
        "actions_opponent": actions1,
        "stderr_tail_hunter": stderr_tails[0],
        "stderr_tail_opponent": stderr_tails[1],
    }


def verify_replay(doc: dict) -> dict:
    """Deterministic replay of the transcript through the pinned engine."""
    env = GeneralsEnv(mode="competition")
    state = make_board(env, doc["seed"])
    transition = make_transition(env)
    winner = -1
    turn = 0
    for a0, a1 in zip(doc["actions_hunter"], doc["actions_opponent"]):
        state, info = transition(state, jnp.stack([jnp.array(a0, dtype=jnp.int32),
                                                   jnp.array(a1, dtype=jnp.int32)]))
        turn += 1
        if bool(info.is_done):
            winner = int(info.winner)
            break
    return {
        "replay_winner": winner,
        "replay_turns": turn,
        "replay_match": winner == doc["winner"] and turn == doc["turns"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="20260901-20260920")
    args = parser.parse_args()
    lo, hi = (int(x) for x in args.seeds.split("-"))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    wins = engine_faults = 0
    docs = []
    for seed in range(lo, hi + 1):
        stamp = time.strftime("%H:%M:%SZ", time.gmtime())
        doc = play_recorded(seed)
        doc.update(verify_replay(doc))
        docs.append(doc)
        hunter_win = doc["winner"] == 0 and not doc["truncated"]
        wins += int(hunter_win)
        engine_faults += doc["faults_hunter"] + doc["faults_opponent"]
        print(
            f"{stamp} seed={seed} winner={doc['winner']} turns={doc['turns']} "
            f"truncated={doc['truncated']} faults={doc['faults_hunter']}/{doc['faults_opponent']} "
            f"replay_match={doc['replay_match']} ({doc['elapsed_s']:.1f}s)",
            flush=True,
        )

    import hashlib

    transcript_path = OUT_ROOT / "transcripts.json"
    payload = json.dumps(docs, sort_keys=True).encode()
    transcript_path.write_bytes(payload)
    summary = {
        "plan": "experiments/marathon/stage5_teacher_r1_plan.yaml",
        "experiment_id": "experiment#stage5-teacher-r1#153b464617b7",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": list(range(lo, hi + 1)),
        "hunter_wins": wins,
        "games": len(docs),
        "engine_faults": engine_faults,
        "all_replays_match": all(d["replay_match"] for d in docs),
        "transcript_sha256": hashlib.sha256(payload).hexdigest(),
        "gate_predeclared": "hunter_wins >= 16/20 AND engine_faults == 0 AND all_replays_match",
        "gate_pass": wins >= 16 and engine_faults == 0
        and all(d["replay_match"] for d in docs),
    }
    (OUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, indent=1), flush=True)
    print("GATE_PASS" if summary["gate_pass"] else "GATE_FAIL", flush=True)
    return 0 if summary["gate_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
