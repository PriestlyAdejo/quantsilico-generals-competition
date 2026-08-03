"""Behaviour cloning smoke tests."""

from __future__ import annotations

from pathlib import Path

import torch

from generals_bot.models.checkpoint import apply_state_dict
from generals_bot.models.factory import build_model
from generals_bot.models.legal_mask import legal_mask_observation, select_legal_action
from generals_bot.observation import Observation
from generals_bot.training.behaviour_clone import run_bc_pipeline


def test_bc_tiny_overfits_mlp(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(Path.cwd())
    summary = run_bc_pipeline(
        tiny=True,
        architectures=["recurrent_mlp_v1"],
        epochs=30,
    )
    acc = summary["reports"]["recurrent_mlp_v1"]["final_train_action_acc"]
    assert acc >= 0.7
    ckpt = Path(summary["reports"]["recurrent_mlp_v1"]["checkpoint"])
    model = build_model("recurrent_mlp_v1")
    apply_state_dict(model, ckpt, map_location="cpu")
    model.eval()
    obs = Observation(
        3,
        3,
        1,
        1,
        5,
        1,
        3,
        ((4, 1, 1), (1, 2, 1), (1, 1, 1)),
        ((1, 1, 0), (0, 0, 0), (0, 0, 2)),
        ((5, 2, 0), (0, 0, 0), (0, 0, 3)),
    )
    with torch.inference_mode():
        logits, _, _ = model.forward_obs(obs, model.initial_hidden())
        idx = select_legal_action(logits, legal_mask_observation(obs))
    assert legal_mask_observation(obs)[idx]
