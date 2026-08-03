"""Factorised action head unit tests."""

from __future__ import annotations

from pathlib import Path

import torch

from generals_bot.models.action_index import ACTION_DIM, action_to_index, index_to_action
from generals_bot.models.checkpoint import apply_state_dict, save_checkpoint
from generals_bot.models.factory import build_model
from generals_bot.models.factorised_action import (
    compose_action_index,
    decompose_action_index,
    roundtrip_ok,
)
from generals_bot.models.legal_mask import legal_mask_observation, select_legal_action
from generals_bot.observation import Observation
from generals_bot.action import Action


def _obs() -> Observation:
    return Observation(
        4,
        5,
        10,
        2,
        12,
        1,
        4,
        ((4, 1, 1, 1, 1), (1, 2, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1)),
        ((1, 1, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 2)),
        ((8, 3, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 4)),
    )


def test_factorised_roundtrip_pass_build_move() -> None:
    samples = [
        Action.pass_(),
        Action.build(2, 3),
        Action.move(1, 1, 0, split=0),
        Action.move(0, 0, 3, split=1),
    ]
    for action in samples:
        idx = action_to_index(action)
        assert roundtrip_ok(idx)
        parts = decompose_action_index(idx)
        rebuilt = compose_action_index(
            action_type=parts["action_type"],
            source=parts["source"],
            direction=parts["direction"],
            split=parts["split"],
        )
        assert rebuilt == idx
        assert index_to_action(rebuilt) == action


def test_cnn_v2_factorised_shapes_and_legal() -> None:
    model = build_model("recurrent_cnn_v2").eval()
    assert model.config.schema_version == 2  # type: ignore[attr-defined]
    obs = _obs()
    with torch.inference_mode():
        logits, value, h = model.forward_obs(obs, model.initial_hidden())
    assert logits.shape == (1, ACTION_DIM)
    assert value.shape == (1,)
    idx = select_legal_action(logits, legal_mask_observation(obs))
    assert bool(legal_mask_observation(obs)[idx])
    # Spatial parameter sharing: source head is Conv2d 1x1
    assert hasattr(model.actor, "source_head")


def test_graph_v2_factorised_shapes() -> None:
    model = build_model("recurrent_graph_belief_v2").eval()
    obs = _obs()
    with torch.inference_mode():
        logits, value, h, cell = model.forward_obs(obs, model.initial_hidden())
    assert logits.shape[-1] == ACTION_DIM
    assert cell.ndim == 4


def test_mlp_labelled_flat_limitation() -> None:
    model = build_model("recurrent_mlp_v1")
    assert "flat" in model.config.action_head  # type: ignore[attr-defined]
    assert "flat absolute" in model.config.limitation  # type: ignore[attr-defined]


def test_v2_safetensors_and_official_cpu_load(tmp_path: Path) -> None:
    model = build_model("recurrent_cnn_v2")
    path = tmp_path / "cnn_v2"
    save_checkpoint(model, path, architecture="recurrent_cnn_v2", config=model.config_dict())
    loaded = build_model("recurrent_cnn_v2")
    apply_state_dict(loaded, path.with_suffix(".json"), map_location="cpu")
    obs = _obs()
    model.eval()
    loaded.eval()
    with torch.inference_mode():
        a, _, _ = model.forward_obs(obs, model.initial_hidden())
        b, _, _ = loaded.forward_obs(obs, loaded.initial_hidden())
    assert torch.allclose(a, b, atol=1e-5)
