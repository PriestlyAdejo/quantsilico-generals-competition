"""MLP model unit tests."""

from __future__ import annotations

import torch

from generals_bot.models.mlp import RecurrentMLPPolicy
from generals_bot.observation import Observation


def _obs() -> Observation:
    return Observation(
        height=3,
        width=3,
        turn=10,
        my_land=2,
        my_army=10,
        opp_land=1,
        opp_army=5,
        type_grid=((4, 1, 1), (1, 2, 1), (1, 1, 1)),
        owner_grid=((1, 1, 0), (0, 0, 0), (0, 0, 2)),
        army_grid=((5, 3, 0), (0, 0, 0), (0, 0, 4)),
    )


def test_mlp_forward_cpu() -> None:
    model = RecurrentMLPPolicy()
    hidden = model.initial_hidden()
    logits, value, new_h = model.forward_obs(_obs(), hidden)
    assert logits.shape[-1] > 1
    assert value.shape == (1,)
    assert new_h.shape == hidden.shape
    assert torch.isfinite(logits).all()


def test_mlp_deterministic() -> None:
    model = RecurrentMLPPolicy()
    model.eval()
    h = model.initial_hidden()
    with torch.no_grad():
        a1, _, h1 = model.forward_obs(_obs(), h)
        a2, _, _ = model.forward_obs(_obs(), h)
    assert torch.allclose(a1, a2)
    assert torch.isfinite(h1).all()
