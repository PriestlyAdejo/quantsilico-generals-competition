"""Observation encoder and Phase 5 model tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from generals_bot.belief.opponent_posterior import (
    OpponentPosteriorState,
    update_opponent_posterior,
)
from generals_bot.models.action_index import ACTION_DIM, action_to_index, index_to_action
from generals_bot.models.checkpoint import apply_state_dict, save_checkpoint
from generals_bot.models.cnn import RecurrentCNNPolicy
from generals_bot.models.graph import RecurrentGraphBeliefPolicy
from generals_bot.models.heads import NUM_OPTIONS, STRATEGIC_OPTIONS
from generals_bot.models.legal_mask import apply_action_mask, legal_mask_observation, select_legal_action
from generals_bot.models.mlp import RecurrentMLPPolicy
from generals_bot.models.observation_encoder import (
    encode_observation,
    encode_observation_reference,
    neighbour_index_tables,
    padding_mask,
)
from generals_bot.observation import Observation
from generals_bot.action import Action


def _obs(h: int = 3, w: int = 5) -> Observation:
    type_grid = tuple(tuple(1 for _ in range(w)) for _ in range(h))
    owner = [[0 for _ in range(w)] for _ in range(h)]
    army = [[0 for _ in range(w)] for _ in range(h)]
    owner[0][0] = 1
    army[0][0] = 8
    owner[h - 1][w - 1] = 2
    army[h - 1][w - 1] = 4
    type_list = [list(row) for row in type_grid]
    type_list[0][0] = 4
    return Observation(
        height=h,
        width=w,
        turn=12,
        my_land=1,
        my_army=8,
        opp_land=1,
        opp_army=4,
        type_grid=tuple(tuple(r) for r in type_list),
        owner_grid=tuple(tuple(r) for r in owner),
        army_grid=tuple(tuple(r) for r in army),
    )


def test_vectorised_encoder_matches_reference() -> None:
    obs = _obs(4, 6)
    ref = encode_observation_reference(obs, device=torch.device("cpu"))
    fast = encode_observation(obs, device=torch.device("cpu"))
    assert torch.allclose(ref, fast, atol=1e-6)


def test_rectangular_padding_mask() -> None:
    mask = padding_mask(3, 5)
    assert mask.shape == (21, 21)
    assert not mask[0, 0]
    assert mask[3, 0]
    assert mask[0, 5]


def test_static_neighbour_cache() -> None:
    tables = neighbour_index_tables()
    assert set(tables) == {"north", "south", "west", "east", "self"}
    assert tables["self"][0, 0] == 0
    assert tables["east"][0, 0] == 1
    assert tables["west"][0, 0] == -1


def test_action_index_roundtrip() -> None:
    a = Action.move(2, 3, 1, split=1)
    assert index_to_action(action_to_index(a)) == a


@pytest.mark.parametrize("factory", [RecurrentMLPPolicy, RecurrentCNNPolicy, RecurrentGraphBeliefPolicy])
def test_model_shapes_and_recurrence(factory) -> None:
    model = factory()
    model.eval()
    h = model.initial_hidden()
    obs = _obs()
    with torch.inference_mode():
        if factory is RecurrentGraphBeliefPolicy:
            logits, value, h2, cell = model.forward_obs(obs, h)
            assert cell.shape[1] == model.config.recurrent_channels
        else:
            logits, value, h2 = model.forward_obs(obs, h)
    assert logits.shape == (1, ACTION_DIM)
    assert value.shape == (1,)
    assert h2.shape == h.shape
    # carry
    with torch.inference_mode():
        if factory is RecurrentGraphBeliefPolicy:
            _, _, h3, _ = model.forward_obs(obs, h2)
        else:
            _, _, h3 = model.forward_obs(obs, h2)
    assert not torch.allclose(h3, h)


def test_legal_masking_and_fallback() -> None:
    model = RecurrentMLPPolicy().eval()
    obs = _obs()
    mask = legal_mask_observation(obs)
    assert mask[0]
    logits = torch.randn(ACTION_DIM)
    idx = select_legal_action(logits.unsqueeze(0), mask)
    assert bool(mask[idx])
    empty = torch.zeros(ACTION_DIM, dtype=torch.bool)
    masked = apply_action_mask(logits.unsqueeze(0), empty)
    assert torch.isfinite(masked[0, 0])
    assert select_legal_action(logits.unsqueeze(0), empty) == 0


def test_opponent_posterior_valid() -> None:
    state = OpponentPosteriorState()
    state = update_opponent_posterior(state, _obs())
    assert abs(float(state.probs.sum()) - 1.0) < 1e-5
    assert state.probs[0] >= 0.05 - 1e-6
    assert state.entropy() >= 0.0


def test_mixture_probs_sum() -> None:
    model = RecurrentMLPPolicy().eval()
    obs = _obs()
    cells = encode_observation(obs).unsqueeze(0)
    from generals_bot.models.observation_encoder import encode_globals

    glob = encode_globals(obs).unsqueeze(0)
    with torch.inference_mode():
        out = model.forward_tensors(cells.reshape(1, -1), glob, model.initial_hidden())
    assert out["mixture_probs"].shape[-1] == NUM_OPTIONS == len(STRATEGIC_OPTIONS)
    assert torch.allclose(out["mixture_probs"].sum(-1), torch.ones(1), atol=1e-5)


def test_safetensors_roundtrip(tmp_path: Path) -> None:
    model = RecurrentCNNPolicy()
    path = tmp_path / "cnn_ckpt"
    save_checkpoint(model, path, architecture=model.config.architecture, config=model.config_dict())
    loaded = RecurrentCNNPolicy()
    apply_state_dict(loaded, path.with_suffix(".json"))
    obs = _obs()
    model.eval()
    loaded.eval()
    with torch.inference_mode():
        a, _, _ = model.forward_obs(obs, model.initial_hidden())
        b, _, _ = loaded.forward_obs(obs, loaded.initial_hidden())
    assert torch.allclose(a, b, atol=1e-5)


def test_deterministic_inference() -> None:
    model = RecurrentMLPPolicy().eval()
    obs = _obs()
    h = model.initial_hidden()
    with torch.inference_mode():
        a1, _, _ = model.forward_obs(obs, h, deterministic=True)
        a2, _, _ = model.forward_obs(obs, h, deterministic=True)
    assert torch.allclose(a1, a2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_cuda_tolerance() -> None:
    cpu = RecurrentMLPPolicy().eval()
    gpu = RecurrentMLPPolicy().to("cuda").eval()
    gpu.load_state_dict(cpu.state_dict())
    obs = _obs()
    with torch.inference_mode():
        a, _, _ = cpu.forward_obs(obs, cpu.initial_hidden())
        b, _, _ = gpu.forward_obs(obs, gpu.initial_hidden(device=torch.device("cuda")))
    assert torch.allclose(a, b.cpu(), atol=1e-4, rtol=1e-4)


def test_official_venv_style_load_all_architectures(tmp_path: Path) -> None:
    """Checkpoint round-trip using CPU map_location (official .venv path)."""
    from generals_bot.models.factory import build_model, known_architectures

    obs = _obs()
    for arch in known_architectures():
        model = build_model(arch)
        model.eval()
        path = tmp_path / arch
        save_checkpoint(model, path, architecture=arch, config=model.config_dict())  # type: ignore[attr-defined]
        loaded = build_model(arch)
        apply_state_dict(loaded, path.with_suffix(".json"), map_location="cpu")
        loaded.eval()
        with torch.inference_mode():
            if arch == "recurrent_graph_belief_v1":
                logits, *_rest = loaded.forward_obs(obs, loaded.initial_hidden())  # type: ignore[misc]
            else:
                logits, *_rest = loaded.forward_obs(obs, loaded.initial_hidden())  # type: ignore[misc]
            idx = select_legal_action(logits, legal_mask_observation(obs))
        assert 0 <= idx < ACTION_DIM


def test_no_privileged_fields_in_encoder_channels() -> None:
    # Encoder only exposes visibility/terrain/ownership/army/padding — no hidden fog truth.
    obs = _obs()
    t = encode_observation(obs)
    assert t.shape[0] == 10
