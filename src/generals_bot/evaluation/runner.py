"""Live-like competition match runner with timing, faults and telemetry."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import psutil
from generals import GeneralsEnv
from generals.core import game

from generals_bot.evaluation.match import make_board, make_transition
from generals_bot.evaluation.metrics import summarize_latencies
from generals_bot.evaluation.replay import write_replay_summary
from generals_bot.rules import ACTION_DEADLINE_S, FAULT_BUDGET, FIRST_ACTION_DEADLINE_S
from generals_bot.schemas import SCHEMA_VERSION, MatchResultRecord

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_COMPETITION = REPO_ROOT / "third_party" / "generals-bots" / "competition"


def _load_engine_protocol():
    import importlib.util

    protocol_path = ENGINE_COMPETITION / "protocol.py"
    spec = importlib.util.spec_from_file_location(
        "generals_competition_protocol", protocol_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load engine protocol from {protocol_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class AgentTelemetry:
    latencies_ms: list[float] = field(default_factory=list)
    protocol_faults: int = 0
    illegal_actions: int = 0
    crash: bool = False
    forfeited: bool = False
    peak_memory_mb: float = 0.0
    classifications: list[str] = field(default_factory=list)


def _read_line_with_timeout(
    proc: subprocess.Popen[str],
    timeout_s: float,
) -> tuple[str | None, str]:
    """Return (line, classification). classification is ok|timeout|eof|crash."""
    if proc.poll() is not None:
        return None, "crash"
    assert proc.stdout is not None
    result_q: queue.Queue[str | None] = queue.Queue()

    def _worker() -> None:
        try:
            line = proc.stdout.readline()
            result_q.put(line if line else None)
        except Exception:
            result_q.put(None)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)
    if thread.is_alive():
        return None, "timeout"
    line = result_q.get_nowait() if not result_q.empty() else None
    if line is None:
        if proc.poll() is not None:
            return None, "crash"
        return None, "eof"
    return line, "ok"


def _sample_memory_mb(proc: subprocess.Popen[str]) -> float:
    try:
        if proc.pid is None:
            return 0.0
        return psutil.Process(proc.pid).memory_info().rss / (1024 * 1024)
    except (psutil.Error, ValueError):
        return 0.0


def run_live_like_match(
    agent0_main: Path,
    agent1_main: Path,
    *,
    seed: int = 0,
    mode: str = "competition",
    max_turns: int | None = None,
    enforce_deadlines: bool = True,
    record_dir: Path | None = None,
    experiment_id: str = "",
    candidate: str = "",
    opponent: str = "",
    seed_split: str | None = None,
) -> MatchResultRecord:
    """Run a competition-mode match with live-like fault accounting."""
    protocol = _load_engine_protocol()
    encode_handshake = protocol.encode_handshake
    encode_observation = protocol.encode_observation
    decode_action = protocol.decode_action

    env = GeneralsEnv(mode=mode)
    get_obs = game.get_full_observation if env.perfect_info else game.get_observation
    state = make_board(env, seed)
    height, width = (int(d) for d in state.armies.shape)
    transition = make_transition(env)
    truncation = int(env.truncation if max_turns is None else min(env.truncation, max_turns))

    def spawn(main_path: Path, player_id: int) -> subprocess.Popen[str]:
        env_vars = os.environ.copy()
        src = str(REPO_ROOT / "src")
        env_vars["PYTHONPATH"] = src + os.pathsep + env_vars.get("PYTHONPATH", "")
        proc = subprocess.Popen(
            [os.environ.get("PYTHON", None) or __import__("sys").executable, "-u", str(main_path)],
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

    p0 = spawn(agent0_main, 0)
    p1 = spawn(agent1_main, 1)
    tele = [AgentTelemetry(), AgentTelemetry()]
    telemetry_rows: list[dict[str, Any]] = []
    first = [True, True]
    turn = 0
    winner = -1
    truncated = False
    t0 = time.perf_counter()

    def ask(proc: subprocess.Popen[str], obs, idx: int) -> jnp.ndarray:
        assert proc.stdin is not None
        deadline = FIRST_ACTION_DEADLINE_S if first[idx] else ACTION_DEADLINE_S
        first[idx] = False
        started = time.perf_counter()
        try:
            proc.stdin.write(encode_observation(obs))
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            tele[idx].crash = True
            tele[idx].protocol_faults += 1
            tele[idx].classifications.append("crash")
            return jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)

        timeout = deadline if enforce_deadlines else max(deadline, 30.0)
        line, classification = _read_line_with_timeout(proc, timeout)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        tele[idx].latencies_ms.append(elapsed_ms)
        tele[idx].peak_memory_mb = max(tele[idx].peak_memory_mb, _sample_memory_mb(proc))

        if classification != "ok" or line is None:
            tele[idx].protocol_faults += 1
            tele[idx].classifications.append(classification)
            if classification == "crash":
                tele[idx].crash = True
            if tele[idx].protocol_faults >= FAULT_BUDGET or tele[idx].crash:
                tele[idx].forfeited = True
            return jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)

        # Extra lines: if another line is already waiting without a new obs, count fault.
        # We only detect obvious multi-line bursts by non-blocking poll of leftover.
        stripped = line.strip()
        parts = stripped.split()
        if len(parts) != 5:
            tele[idx].protocol_faults += 1
            tele[idx].classifications.append("malformed")
            return jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
        try:
            action = decode_action(stripped)
        except Exception:
            tele[idx].protocol_faults += 1
            tele[idx].classifications.append("malformed")
            return jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)

        # Illegal-but-well-formed actions are silent passes at engine level;
        # we still count them separately when clearly impossible.
        kind = int(action[0])
        if kind not in (0, 1, 2):
            tele[idx].illegal_actions += 1
            tele[idx].classifications.append("illegal_action")
            return jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)

        if enforce_deadlines and elapsed_ms > deadline * 1000.0:
            tele[idx].protocol_faults += 1
            tele[idx].classifications.append("late")
            return jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)

        tele[idx].classifications.append("ok")
        return action

    try:
        while turn < truncation:
            if tele[0].forfeited or tele[1].forfeited:
                winner = 1 if tele[0].forfeited and not tele[1].forfeited else (
                    0 if tele[1].forfeited and not tele[0].forfeited else -1
                )
                break
            obs0 = get_obs(state, 0)
            obs1 = get_obs(state, 1)
            a0 = ask(p0, obs0, 0)
            a1 = ask(p1, obs1, 1)
            state, info = transition(state, jnp.stack([a0, a1]))
            turn += 1
            telemetry_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "turn": turn,
                    "faults0": tele[0].protocol_faults,
                    "faults1": tele[1].protocol_faults,
                    "latency_ms0": tele[0].latencies_ms[-1] if tele[0].latencies_ms else None,
                    "latency_ms1": tele[1].latencies_ms[-1] if tele[1].latencies_ms else None,
                }
            )
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

    lat0 = summarize_latencies(tele[0].latencies_ms)
    lat1 = summarize_latencies(tele[1].latencies_ms)
    record = MatchResultRecord(
        experiment_id=experiment_id,
        seed=seed,
        candidate=candidate or str(agent0_main),
        opponent=opponent or str(agent1_main),
        winner=winner,
        turns=turn,
        faults0=tele[0].protocol_faults,
        faults1=tele[1].protocol_faults,
        crash0=tele[0].crash,
        crash1=tele[1].crash,
        forfeited0=tele[0].forfeited,
        forfeited1=tele[1].forfeited,
        truncated=truncated,
        elapsed_s=time.perf_counter() - t0,
        peak_memory_mb0=tele[0].peak_memory_mb or None,
        peak_memory_mb1=tele[1].peak_memory_mb or None,
        latency_p50_ms0=lat0["p50"],
        latency_p99_ms0=lat0["p99"],
        latency_p50_ms1=lat1["p50"],
        latency_p99_ms1=lat1["p99"],
        illegal_action_count0=tele[0].illegal_actions,
        illegal_action_count1=tele[1].illegal_actions,
        protocol_fault_count0=tele[0].protocol_faults,
        protocol_fault_count1=tele[1].protocol_faults,
        seed_split=seed_split,
        notes=[
            f"classifications0={tele[0].classifications[:8]}",
            f"classifications1={tele[1].classifications[:8]}",
        ],
    )

    if record_dir is not None:
        record_dir.mkdir(parents=True, exist_ok=True)
        replay_path = record_dir / f"match_s{seed}_{candidate}_vs_{opponent}.json"
        telem_path = record_dir / f"match_s{seed}_{candidate}_vs_{opponent}.jsonl"
        write_replay_summary(replay_path, record.to_dict())
        with telem_path.open("w", encoding="utf-8") as fh:
            for row in telemetry_rows:
                fh.write(json.dumps(row) + "\n")
        record.replay_path = str(replay_path)
        record.telemetry_path = str(telem_path)
        (record_dir / f"match_s{seed}_result.json").write_text(
            json.dumps(asdict(record), indent=2) + "\n",
            encoding="utf-8",
        )
    return record
