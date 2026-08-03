"""Minimal Windows-safe subprocess match runner (no bare bash)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import jax.random as jrandom
from generals import GeneralsEnv
from generals.core import game
from generals.core.game import create_initial_state
from generals.core.grid import generate_grid
from generals.modifiers import build_castles as _bc
from generals.modifiers import deathtouch as _dt

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_COMPETITION = REPO_ROOT / "third_party" / "generals-bots" / "competition"


@dataclass
class MatchResult:
    winner: int
    turns: int
    seed: int
    agent0: str
    agent1: str
    faults0: int = 0
    faults1: int = 0
    elapsed_s: float = 0.0
    truncated: bool = False


def make_board(env: GeneralsEnv, seed: int):
    """Mirror competition/matchup.py make_board."""
    key = jrandom.PRNGKey(seed)
    if env._fixed_dims is not None:
        return env.init_state(key)

    kd, kg = jrandom.split(key)
    lo, hi = env.min_grid_size, env.max_grid_size
    h = int(jrandom.randint(kd, (), lo, hi + 1))
    w = int(jrandom.randint(jrandom.fold_in(kd, 1), (), lo, hi + 1))
    grid = generate_grid(
        kg,
        grid_dims=(h, w),
        mountain_density_range=env.mountain_density_range,
        num_castles_range=env.num_castles_range,
        min_generals_distance=env.min_generals_distance,
        castle_val_range=env.castle_val_range,
    )[:h, :w]
    if env.build_castles:
        grid = _bc.strip_neutral_castles(grid)
    return create_initial_state(grid.astype(jnp.int32))


def make_transition(env: GeneralsEnv):
    """Mirror competition/matchup.py make_transition."""

    def transition(state, actions):
        if env.build_castles:
            state, actions = _bc.apply_build_actions(state, actions)
        if env.deathtouch_turn is not None:
            return _dt.step(state, actions, env.deathtouch_turn)
        return game.step(state, actions)

    return transition


def _load_engine_protocol():
    """Load competition/protocol.py by path (avoids package name clashes)."""
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


def run_python_agent_match(
    agent0_main: Path,
    agent1_main: Path,
    *,
    seed: int = 0,
    mode: str = "competition",
    max_turns: int | None = None,
) -> MatchResult:
    """Spawn two ``python -u main.py`` agents and play a competition match."""
    protocol = _load_engine_protocol()
    decode_action = protocol.decode_action
    encode_handshake = protocol.encode_handshake
    encode_observation = protocol.encode_observation

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

    p0 = spawn(agent0_main, 0)
    p1 = spawn(agent1_main, 1)
    faults = [0, 0]
    t0 = time.perf_counter()
    turn = 0
    winner = -1
    truncated = False

    def ask(proc: subprocess.Popen[str], obs, idx: int) -> jnp.ndarray:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(encode_observation(obs))
        proc.stdin.flush()
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
            a0 = ask(p0, obs0, 0)
            a1 = ask(p1, obs1, 1)
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

    return MatchResult(
        winner=winner,
        turns=turn,
        seed=seed,
        agent0=str(agent0_main),
        agent1=str(agent1_main),
        faults0=faults[0],
        faults1=faults[1],
        elapsed_s=time.perf_counter() - t0,
        truncated=truncated,
    )
