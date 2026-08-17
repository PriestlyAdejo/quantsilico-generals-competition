"""OBS-V2 training/serving parity fixture (obs_v2_r1_plan.yaml parity mandate).

For identical engine states, the OBS-V2 tensors the protocol serving path
derives (parse_observation_frame -> encode_observation_v2) must equal the
training-path tensors (observe_one_v2), with memory accumulation included.
The serving path is exercised through the pinned engine's official protocol
encoder, exactly as run_python_agent_match delivers it. Both paths consume
only fog-applied legal observation content.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from generals.core import game

from generals_bot.competition_native_jax.competition_env_jax import (
    index_to_engine_action,
    legal_mask_one_p0,
    legal_mask_one_p1,
    reset_one_jax,
    step_one_jax,
)
from generals_bot.competition_native_jax.obs_memory import (
    ObsMemoryV2,
    encode_observation_v2,
)
from generals_bot.competition_native_jax.obs_v2_jax import (
    N_GLOBAL_V2,
    N_SPATIAL_V2,
    empty_memory_v2,
    observe_one_v2,
)
from generals_bot.protocol import parse_observation_frame

REPO = Path(__file__).resolve().parents[2]
ENGINE_PROTOCOL = REPO / "third_party/generals-bots/competition/protocol.py"


def _engine_protocol():
    spec = importlib.util.spec_from_file_location(
        "generals_competition_protocol_obs_v2_parity", ENGINE_PROTOCOL
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _host_observation(protocol, state, player: int):
    obs = game.get_observation(state, player)
    frame = protocol.encode_observation(obs)
    lines = frame.splitlines()
    width = len(lines[1].split())
    height = (len(lines) - 1) // 3
    return parse_observation_frame(lines[0], lines[1:], height=height, width=width)


def _board_state(seed: int, h: int, w: int):
    import jax

    return reset_one_jax(jax.random.PRNGKey(seed), h, w)


def _assert_parity(host_spatial, host_global, train_spatial, train_global, label: str):
    assert np.asarray(train_spatial).shape == (N_SPATIAL_V2, 21, 21)
    assert np.asarray(train_global).shape == (N_GLOBAL_V2,)
    assert np.allclose(host_spatial, np.asarray(train_spatial), atol=1e-6), (
        f"{label} spatial mismatch: "
        f"max_abs_diff={np.abs(host_spatial - np.asarray(train_spatial)).max()}"
    )
    assert np.allclose(host_global, np.asarray(train_global), atol=1e-6), (
        f"{label} global mismatch: host={host_global} "
        f"train={np.asarray(train_global)}"
    )


@pytest.mark.parametrize("seed,h,w", [(11, 18, 19), (23, 21, 21), (37, 19, 18)])
def test_obs_v2_parity_initial(seed: int, h: int, w: int) -> None:
    protocol = _engine_protocol()
    state = _board_state(seed, h, w)
    for player in (0, 1):
        host_obs = _host_observation(protocol, state, player)
        host_spatial, host_global = encode_observation_v2(host_obs, ObsMemoryV2())
        train_spatial, train_global, _ = observe_one_v2(state, player, empty_memory_v2())
        _assert_parity(
            host_spatial, host_global, train_spatial, train_global,
            f"initial seed {seed} player {player}",
        )


def _first_legal_engine_action(mask) -> jnp.ndarray:
    """Deterministic legal engine action: lowest non-PASS index, else PASS."""
    legal = np.flatnonzero(np.asarray(mask))
    moves = legal[legal > 0]
    idx = int(moves[0]) if moves.size else 0
    return index_to_engine_action(jnp.asarray(idx, dtype=jnp.int32))


def test_obs_v2_parity_across_steps_and_memory() -> None:
    """Parity must hold after gameplay changes state and v2 memory accumulates
    (seen_own/last_army plus the enemy-seen and revealed-enemy-general planes).
    """
    protocol = _engine_protocol()
    state = _board_state(41, 18, 20)
    mem_host = {0: ObsMemoryV2(), 1: ObsMemoryV2()}
    mem_jax = {0: empty_memory_v2(), 1: empty_memory_v2()}
    for step in range(6):
        action_pair = jnp.stack(
            [
                _first_legal_engine_action(legal_mask_one_p0(state)),
                _first_legal_engine_action(legal_mask_one_p1(state)),
            ]
        )
        state, _, _, _, info = step_one_jax(state, action_pair)
        if bool(info.is_done):
            break
        for player in (0, 1):
            host_obs = _host_observation(protocol, state, player)
            host_spatial, host_global = encode_observation_v2(host_obs, mem_host[player])
            train_spatial, train_global, new_mem = observe_one_v2(
                state, player, mem_jax[player]
            )
            mem_jax[player] = new_mem
            _assert_parity(
                host_spatial, host_global, train_spatial, train_global,
                f"step {step} player {player}",
            )
