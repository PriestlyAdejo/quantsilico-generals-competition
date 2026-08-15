"""Observation/mask parity: host protocol path == training path (EVAL_ONLY).

For identical engine states, the features a protocol agent derives from the
stdio observation stream (encode_observation + legal_mask_from_observation)
must equal the training-time tensors (observe_one_jax + legal_mask_one_jax).
The serving path is exercised through the pinned engine's official protocol
encoder and the host parser, exactly as run_python_agent_match delivers it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from generals.core import game

from generals_bot.competition_native_jax.competition_env_jax import (
    empty_memory,
    index_to_engine_action,
    legal_mask_one_p0,
    legal_mask_one_p1,
    observe_one_jax,
    reset_one_jax,
    step_one_jax,
)
from generals_bot.competition_native_jax.legal_mask import legal_mask_from_observation
from generals_bot.competition_native_jax.obs_memory import ObsMemory, encode_observation
from generals_bot.protocol import parse_observation_frame

REPO = Path(__file__).resolve().parents[2]
ENGINE_PROTOCOL = REPO / "third_party/generals-bots/competition/protocol.py"


def _engine_protocol():
    spec = importlib.util.spec_from_file_location(
        "generals_competition_protocol_parity", ENGINE_PROTOCOL
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _host_observation(protocol, state, player: int):
    """Reproduce the exact serving path: official encode -> host parse.

    Height/width are read from the encoded frame itself (rectangular resets
    may pad the observation to the square pad_to), matching how the match
    runner trusts the wire frame.
    """
    obs = game.get_observation(state, player)
    frame = protocol.encode_observation(obs)
    lines = frame.splitlines()
    width = len(lines[1].split())
    height = (len(lines) - 1) // 3
    return parse_observation_frame(lines[0], lines[1:], height=height, width=width)


def _board_state(seed: int, h: int, w: int):
    import jax

    return reset_one_jax(jax.random.PRNGKey(seed), h, w)


@pytest.mark.parametrize("seed,h,w", [(11, 18, 19), (23, 21, 21), (37, 19, 18)])
def test_observation_parity_initial(seed: int, h: int, w: int) -> None:
    protocol = _engine_protocol()
    state = _board_state(seed, h, w)
    for player, legal_mask_one in ((0, legal_mask_one_p0), (1, legal_mask_one_p1)):
        host_obs = _host_observation(protocol, state, player)
        host_spatial, host_global = encode_observation(host_obs, ObsMemory())
        train_spatial, train_global, _ = observe_one_jax(state, player, empty_memory())
        assert np.allclose(host_spatial, np.asarray(train_spatial), atol=1e-6), (
            f"spatial mismatch player {player}: "
            f"max_abs_diff={np.abs(host_spatial - np.asarray(train_spatial)).max()}"
        )
        assert np.allclose(host_global, np.asarray(train_global), atol=1e-6), (
            f"global mismatch player {player}: host={host_global} "
            f"train={np.asarray(train_global)}"
        )
        host_mask = legal_mask_from_observation(host_obs)
        train_mask = np.asarray(legal_mask_one(state))
        assert bool(host_mask[0]), "PASS must always be legal"
        assert np.array_equal(host_mask, train_mask), (
            f"mask mismatch player {player}: host_legal={int(host_mask.sum())} "
            f"train_legal={int(train_mask.sum())} "
            f"xor={int(np.logical_xor(host_mask, train_mask).sum())}"
        )


def _first_legal_engine_action(mask) -> jnp.ndarray:
    """Deterministic legal engine action: lowest non-PASS index, else PASS."""
    legal = np.flatnonzero(np.asarray(mask))
    moves = legal[legal > 0]
    idx = int(moves[0]) if moves.size else 0
    return index_to_engine_action(jnp.asarray(idx, dtype=jnp.int32))


def test_observation_parity_across_steps_and_memory() -> None:
    """Parity must hold after gameplay changes state and memory accumulates."""
    protocol = _engine_protocol()
    state = _board_state(41, 18, 20)
    mem_host = {0: ObsMemory(), 1: ObsMemory()}
    mem_jax = {0: empty_memory(), 1: empty_memory()}
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
            host_spatial, host_global = encode_observation(host_obs, mem_host[player])
            train_spatial, train_global, new_mem = observe_one_jax(
                state, player, mem_jax[player]
            )
            mem_jax[player] = new_mem
            assert np.allclose(host_spatial, np.asarray(train_spatial), atol=1e-6), (
                f"step {step} player {player} spatial mismatch: "
                f"max_abs_diff={np.abs(host_spatial - np.asarray(train_spatial)).max()}"
            )
            assert np.allclose(host_global, np.asarray(train_global), atol=1e-6)
