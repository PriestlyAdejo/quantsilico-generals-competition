from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from train.competition_native_jax.deadline_rescue_bc_jax import (
    action_type_logits,
    action_type_targets,
    ranking_loss_per_sample,
    zero_value_gradient,
)
from train.competition_native_jax.ppo_jax import combine_ppo_and_anchor_gradients


def test_action_type_mapping_covers_canonical_3970_indices() -> None:
    targets = np.asarray(action_type_targets(jnp.arange(3_970)))
    assert np.count_nonzero(targets == 0) == 1
    assert np.count_nonzero(targets == 1) == 441 * 8
    assert np.count_nonzero(targets == 2) == 441


def test_action_type_logits_ignore_illegal_high_logit() -> None:
    logits = jnp.zeros((1, 3_970)).at[0, 9].set(1.0e6).at[0, 10].set(3.0)
    legal = jnp.zeros((1, 3_970), dtype=bool).at[0, 0].set(True).at[0, 10].set(True)
    typed = np.asarray(action_type_logits(logits, legal))[0]
    assert typed[0] == 0.0
    assert typed[1] == 3.0
    assert typed[2] < -1.0e20


def test_ranking_loss_uses_only_strongest_wrong_legal_action() -> None:
    legal = jnp.zeros((2, 3_970), dtype=bool)
    legal = legal.at[:, 0].set(True).at[:, 1].set(True).at[:, 2].set(True)
    logits = jnp.zeros((2, 3_970))
    logits = logits.at[0, 1].set(2.0).at[0, 2].set(1.0)
    logits = logits.at[1, 1].set(1.0).at[1, 2].set(2.0)
    loss = np.asarray(ranking_loss_per_sample(logits, legal, jnp.asarray([1, 1])))
    assert loss[0] == 0.0
    assert loss[1] == 1.5


def test_zero_value_gradient_preserves_actor_and_zeros_critic() -> None:
    gradients = {"actor": jnp.ones((2,)), "value_head": jnp.ones((3,))}
    masked = zero_value_gradient(gradients)
    assert np.array_equal(np.asarray(masked["actor"]), np.ones((2,)))
    assert np.array_equal(np.asarray(masked["value_head"]), np.zeros((3,)))


def test_anchor_zero_target_is_exact_ppo_gradient() -> None:
    ppo = {"actor": jnp.asarray([3.0, 4.0]), "value_head": jnp.asarray([2.0])}
    anchor = {"actor": jnp.asarray([10.0, 0.0]), "value_head": jnp.asarray([99.0])}
    combined, metrics = combine_ppo_and_anchor_gradients(
        ppo, anchor, target_ratio=0.0
    )
    assert np.array_equal(np.asarray(combined["actor"]), np.asarray(ppo["actor"]))
    assert np.array_equal(
        np.asarray(combined["value_head"]), np.asarray(ppo["value_head"])
    )
    assert float(metrics["ANCHOR_SCALE"]) == 0.0


def test_tiny_ppo_gradient_cannot_be_disguised_by_anchor() -> None:
    ppo = {"actor": jnp.asarray([1.0e-12]), "value_head": jnp.asarray([2.0])}
    anchor = {"actor": jnp.asarray([10.0]), "value_head": jnp.asarray([99.0])}
    combined, metrics = combine_ppo_and_anchor_gradients(ppo, anchor)
    assert float(metrics["NO_TASK_POLICY_GRADIENT"]) == 1.0
    assert float(metrics["ANCHOR_SCALE"]) == 0.0
    assert np.array_equal(np.asarray(combined["actor"]), np.asarray(ppo["actor"]))
    assert np.array_equal(
        np.asarray(combined["value_head"]), np.asarray(ppo["value_head"])
    )


def test_anchor_gradient_is_actor_only_and_targets_twenty_percent() -> None:
    ppo = {"actor": jnp.asarray([3.0, 4.0]), "value_head": jnp.asarray([2.0])}
    anchor = {"actor": jnp.asarray([10.0, 0.0]), "value_head": jnp.asarray([99.0])}
    combined, metrics = combine_ppo_and_anchor_gradients(ppo, anchor)
    assert np.isclose(float(metrics["ANCHOR_TO_PPO_RATIO"]), 0.2)
    assert np.array_equal(
        np.asarray(combined["value_head"]), np.asarray(ppo["value_head"])
    )
