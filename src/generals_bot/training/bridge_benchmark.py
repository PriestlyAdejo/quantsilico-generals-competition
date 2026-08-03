"""JAX environment ↔ PyTorch policy bridge throughput benchmark."""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch

from generals import GeneralsEnv
from generals.core import game

from generals_bot.evaluation.match import make_board, make_transition
from generals_bot.models.mlp import RecurrentMLPPolicy
from generals_bot.observation import Observation
from generals_bot.protocol import (
    TYPE_CASTLE,
    TYPE_FOG,
    TYPE_GENERAL,
    TYPE_MOUNTAIN,
    TYPE_PLAIN,
    TYPE_STRUCTURE_IN_FOG,
)


def _obs_from_engine(engine_obs, height: int, width: int) -> Observation:
    # Convert engine Observation arrays to our dataclass.
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
    return Observation(
        height=height,
        width=width,
        turn=int(engine_obs.timestep),
        my_land=int(engine_obs.owned_land_count),
        my_army=int(engine_obs.owned_army_count),
        opp_land=int(engine_obs.opponent_land_count),
        opp_army=int(engine_obs.opponent_army_count),
        type_grid=tuple(tuple(int(x) for x in row) for row in type_grid),
        owner_grid=tuple(tuple(int(x) for x in row) for row in owner),
        army_grid=tuple(tuple(int(x) for x in row) for row in armies),
    )


def run_bridge_benchmark(
    *,
    steps: int = 200,
    device: str | None = None,
    seed: int = 0,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    env = GeneralsEnv(mode="competition")
    state = make_board(env, seed)
    transition = make_transition(env)
    get_obs = game.get_observation
    h, w = (int(d) for d in state.armies.shape)
    policy = RecurrentMLPPolicy().to(torch_device)
    policy.eval()
    hidden = policy.initial_hidden(device=torch_device)

    timings = {
        "env_ms": 0.0,
        "extract_ms": 0.0,
        "h2d_ms": 0.0,
        "forward_ms": 0.0,
        "d2h_ms": 0.0,
        "total_ms": 0.0,
    }
    t_all = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            t0 = time.perf_counter()
            eng_obs = get_obs(state, 0)
            timings["env_ms"] += (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            obs = _obs_from_engine(eng_obs, h, w)
            timings["extract_ms"] += (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            # encode happens on device inside forward_obs
            if torch_device.type == "cuda":
                torch.cuda.synchronize()
            timings["h2d_ms"] += (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            logits, _value, hidden = policy.forward_obs(obs, hidden)
            if torch_device.type == "cuda":
                torch.cuda.synchronize()
            timings["forward_ms"] += (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            action_idx = int(torch.argmax(logits, dim=-1).item())
            timings["d2h_ms"] += (time.perf_counter() - t0) * 1000

            # Environment step with pass (benchmark cares about observation path)
            _ = action_idx
            actions = jnp.stack(
                [
                    jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32),
                    jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32),
                ]
            )
            state, info = transition(state, actions)
            if bool(info.is_done):
                state = make_board(env, seed + 1)
                hidden = policy.initial_hidden(device=torch_device)

    timings["total_ms"] = (time.perf_counter() - t_all) * 1000
    steps_per_s = steps / (timings["total_ms"] / 1000.0)
    transfer_pct = 100.0 * (timings["h2d_ms"] + timings["d2h_ms"]) / max(timings["total_ms"], 1e-6)
    decision = "PASS"
    if steps_per_s < 5:
        decision = "FAIL"
    elif steps_per_s < 30:
        decision = "PARTIAL"

    report = {
        "schema_version": 1,
        "device": device,
        "steps": steps,
        "steps_per_second": steps_per_s,
        "games_per_hour_est": steps_per_s * 3600 / 1200,
        "transfer_percentage": transfer_pct,
        "timings_ms": timings,
        "parameter_count": policy.parameter_count(),
        "decision": decision,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    out = Path("experiments/summaries")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "jax_pytorch_bridge_benchmark.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["path"] = str(path)
    return report


def main() -> None:
    report = run_bridge_benchmark()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
