"""JAX PPO loss, ratio identity, and Optax update."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax

from generals_bot.competition_native_jax.constants import (
    HL_GAUSS_BINS,
    HL_GAUSS_MAX,
    HL_GAUSS_MIN,
    HL_GAUSS_SIGMA,
)
from generals_bot.competition_native_jax.inference_jax import masked_log_softmax
from generals_bot.competition_native_jax.transformer_jax import forward


def hl_gauss_target(ret: jax.Array) -> jax.Array:
    centers = jnp.linspace(HL_GAUSS_MIN, HL_GAUSS_MAX, HL_GAUSS_BINS)
    z = jnp.exp(-0.5 * ((centers - ret) / HL_GAUSS_SIGMA) ** 2)
    return z / jnp.maximum(z.sum(), 1e-12)


def policy_ratio(logp_new: jax.Array, logp_old: jax.Array) -> jax.Array:
    return jnp.exp(logp_new - logp_old)


def assert_zero_update_ratio(
    logits: jax.Array,
    mask: jax.Array,
    action: jax.Array,
    atol: float = 1e-5,
) -> jax.Array:
    logp = masked_log_softmax(logits, mask)
    logp_a = logp[action]
    rho = policy_ratio(logp_a, logp_a)
    return rho


def ppo_loss_on_batch(
    params: dict,
    spatial: jax.Array,
    global_vec: jax.Array,
    mask: jax.Array,
    actions: jax.Array,
    old_logp: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
    *,
    clip: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Single-transition or leading-batch axis optional via vmap outside."""

    def one(sp, gv, m, a, olp, adv, ret):
        out = forward(params, sp, gv)
        logp_all = masked_log_softmax(out["flat_logits"], m)
        logp = logp_all[a]
        ratio = jnp.exp(logp - olp)
        unclipped = ratio * adv
        clipped = jnp.clip(ratio, 1.0 - clip, 1.0 + clip) * adv
        pg = -jnp.minimum(unclipped, clipped)
        ent = -jnp.sum(jnp.where(m, jnp.exp(logp_all) * logp_all, 0.0))
        target = hl_gauss_target(ret)
        vloss = -jnp.sum(target * jax.nn.log_softmax(out["value_logits"]))
        loss = pg + vf_coef * vloss - ent_coef * ent
        return loss, {"pg": pg, "vloss": vloss, "entropy": ent, "ratio": ratio}

    losses, metrics = jax.vmap(one)(
        spatial, global_vec, mask, actions, old_logp, advantages, returns
    )
    mean_loss = jnp.mean(losses)
    mean_metrics = {k: jnp.mean(v) for k, v in metrics.items()}
    return mean_loss, mean_metrics


def make_optimizer(lr: float = 3e-4) -> optax.GradientTransformation:
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))


def actor_gradient_norm(gradients: dict) -> jax.Array:
    actor = {key: value for key, value in gradients.items() if key != "value_head"}
    leaves = jax.tree_util.tree_leaves(actor)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def supervised_anchor_loss(
    params: dict,
    spatial: jax.Array,
    global_vec: jax.Array,
    mask: jax.Array,
    actions: jax.Array,
) -> jax.Array:
    def one(sp, gv, legal, action):
        output = forward(params, sp, gv)
        logp = masked_log_softmax(output["flat_logits"], legal)
        return -logp[action]

    return jnp.mean(jax.vmap(one)(spatial, global_vec, mask, actions))


def combine_ppo_and_anchor_gradients(
    ppo_gradients: dict,
    anchor_gradients: dict,
    *,
    target_ratio: float = 0.20,
    epsilon: float = 1e-8,
    maximum_scale: float = 1.0,
) -> tuple[dict, dict[str, jax.Array]]:
    """Combine actor-only anchor gradients without disguising zero PPO signal."""
    anchor_gradients = {
        **anchor_gradients,
        "value_head": jnp.zeros_like(anchor_gradients["value_head"]),
    }
    ppo_norm = actor_gradient_norm(ppo_gradients)
    anchor_norm = actor_gradient_norm(anchor_gradients)
    norms_finite = jnp.isfinite(ppo_norm) & jnp.isfinite(anchor_norm)
    no_task_gradient = (~jnp.isfinite(ppo_norm)) | (ppo_norm < epsilon)
    anchor_degenerate = (~jnp.isfinite(anchor_norm)) | (anchor_norm < epsilon)
    raw_scale = target_ratio * ppo_norm / jnp.maximum(anchor_norm, epsilon)
    scale = jnp.clip(raw_scale, 0.0, maximum_scale)
    scale = jnp.where(
        norms_finite & (~no_task_gradient) & (~anchor_degenerate), scale, 0.0
    )
    combined = jax.tree_util.tree_map(
        lambda ppo, anchor: ppo + scale * anchor,
        ppo_gradients,
        anchor_gradients,
    )
    effective = scale * anchor_norm
    ratio = jnp.where(ppo_norm >= epsilon, effective / ppo_norm, 0.0)
    return combined, {
        "PPO_ACTOR_GRAD_NORM": ppo_norm,
        "ANCHOR_RAW_GRAD_NORM": anchor_norm,
        "ANCHOR_SCALE": scale,
        "ANCHOR_EFFECTIVE_GRAD_NORM": effective,
        "ANCHOR_TO_PPO_RATIO": ratio,
        "NO_TASK_POLICY_GRADIENT": no_task_gradient.astype(jnp.float32),
        "ANCHOR_GRADIENT_DEGENERATE": anchor_degenerate.astype(jnp.float32),
        "ANCHOR_GRADIENTS_FINITE": norms_finite.astype(jnp.float32),
    }


def ppo_update(
    params: dict,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    batch: dict[str, jax.Array],
    *,
    accumulate_minibatches: int | None = None,
) -> tuple[dict, Any, dict]:
    """One logical full-batch Optax update (canonical systems semantics).

    When ``accumulate_minibatches`` is set, split the batch into that many
    static shards, accumulate mean gradients with lax.scan, then apply exactly
    one optimiser step. Parameters are not updated between shards.
    """
    n = int(batch["actions"].shape[0])
    if (
        accumulate_minibatches is None
        or accumulate_minibatches <= 1
        or n % accumulate_minibatches != 0
    ):
        def loss_fn(p):
            loss, metrics = ppo_loss_on_batch(
                p,
                batch["spatial"],
                batch["global"],
                batch["mask"],
                batch["actions"],
                batch["old_logp"],
                batch["advantages"],
                batch["returns"],
            )
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics = {**metrics, "loss": loss}
        return params, opt_state, metrics

    mb = accumulate_minibatches
    shard = n // mb

    def reshape(x):
        return x.reshape((mb, shard) + x.shape[1:])

    shards = {k: reshape(v) for k, v in batch.items()}

    def body(carry, i):
        grad_acc, loss_acc, metrics_acc = carry

        def loss_fn(p):
            loss, metrics = ppo_loss_on_batch(
                p,
                shards["spatial"][i],
                shards["global"][i],
                shards["mask"][i],
                shards["actions"][i],
                shards["old_logp"][i],
                shards["advantages"][i],
                shards["returns"][i],
            )
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grad_acc = jax.tree_util.tree_map(lambda a, b: a + b, grad_acc, grads)
        loss_acc = loss_acc + loss
        metrics_acc = {k: metrics_acc[k] + metrics[k] for k in metrics}
        return (grad_acc, loss_acc, metrics_acc), None

    zero_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    zero_metrics = {"pg": 0.0, "vloss": 0.0, "entropy": 0.0, "ratio": 0.0}
    init = (
        zero_grads,
        jnp.array(0.0, dtype=jnp.float32),
        {key: jnp.array(0.0) for key in zero_metrics},
    )
    (grad_sum, loss_sum, metrics_sum), _ = jax.lax.scan(body, init, jnp.arange(mb))
    grads = jax.tree_util.tree_map(lambda g: g / float(mb), grad_sum)
    loss = loss_sum / float(mb)
    metrics = {k: v / float(mb) for k, v in metrics_sum.items()}
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    metrics = {**metrics, "loss": loss, "accumulate_minibatches": jnp.array(float(mb))}
    return params, opt_state, metrics


ppo_update_jit = jax.jit(ppo_update, static_argnames=("optimizer", "accumulate_minibatches"))


def _ppo_gradients_for_anchor(
    params: dict,
    batch: dict[str, jax.Array],
    *,
    accumulate_minibatches: int | None,
) -> tuple[jax.Array, dict[str, jax.Array], dict]:
    n = int(batch["actions"].shape[0])
    if (
        accumulate_minibatches is None
        or accumulate_minibatches <= 1
        or n % accumulate_minibatches != 0
    ):
        def loss_fn(candidate):
            return ppo_loss_on_batch(
                candidate,
                batch["spatial"],
                batch["global"],
                batch["mask"],
                batch["actions"],
                batch["old_logp"],
                batch["advantages"],
                batch["returns"],
            )

        (loss, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params)
        return loss, metrics, gradients

    minibatches = accumulate_minibatches
    shard = n // minibatches

    def reshape(value):
        return value.reshape((minibatches, shard) + value.shape[1:])

    shards = {key: reshape(value) for key, value in batch.items()}

    def body(carry, index):
        gradient_sum, loss_sum, metric_sum = carry

        def loss_fn(candidate):
            return ppo_loss_on_batch(
                candidate,
                shards["spatial"][index],
                shards["global"][index],
                shards["mask"][index],
                shards["actions"][index],
                shards["old_logp"][index],
                shards["advantages"][index],
                shards["returns"][index],
            )

        (loss, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params)
        gradient_sum = jax.tree_util.tree_map(
            lambda accumulated, current: accumulated + current,
            gradient_sum,
            gradients,
        )
        loss_sum = loss_sum + loss
        metric_sum = {key: metric_sum[key] + metrics[key] for key in metrics}
        return (gradient_sum, loss_sum, metric_sum), None

    zero_gradients = jax.tree_util.tree_map(jnp.zeros_like, params)
    zero_metrics = {
        key: jnp.array(0.0) for key in ("pg", "vloss", "entropy", "ratio")
    }
    initial = (zero_gradients, jnp.array(0.0, dtype=jnp.float32), zero_metrics)
    (gradient_sum, loss_sum, metric_sum), _ = jax.lax.scan(
        body, initial, jnp.arange(minibatches)
    )
    gradients = jax.tree_util.tree_map(
        lambda gradient: gradient / float(minibatches), gradient_sum
    )
    loss = loss_sum / float(minibatches)
    metrics = {key: value / float(minibatches) for key, value in metric_sum.items()}
    metrics["accumulate_minibatches"] = jnp.asarray(float(minibatches))
    return loss, metrics, gradients


def ppo_update_with_anchor(
    params: dict,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    batch: dict[str, jax.Array],
    anchor_batch: dict[str, jax.Array],
    *,
    accumulate_minibatches: int | None = None,
    anchor_target_ratio: float = 0.20,
    anchor_epsilon: float = 1e-8,
    anchor_maximum_scale: float = 1.0,
) -> tuple[dict, Any, dict]:
    ppo_loss, metrics, ppo_gradients = _ppo_gradients_for_anchor(
        params,
        batch,
        accumulate_minibatches=accumulate_minibatches,
    )

    def anchor_objective(candidate):
        return supervised_anchor_loss(
            candidate,
            anchor_batch["spatial"],
            anchor_batch["global"],
            anchor_batch["mask"],
            anchor_batch["actions"],
        )

    anchor_loss, anchor_gradients = jax.value_and_grad(anchor_objective)(params)
    combined, anchor_metrics = combine_ppo_and_anchor_gradients(
        ppo_gradients,
        anchor_gradients,
        target_ratio=anchor_target_ratio,
        epsilon=anchor_epsilon,
        maximum_scale=anchor_maximum_scale,
    )
    updates, opt_state = optimizer.update(combined, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, {
        **metrics,
        **anchor_metrics,
        "loss": ppo_loss,
        "SUPERVISED_ANCHOR_LOSS": anchor_loss,
    }


ppo_update_with_anchor_jit = jax.jit(
    ppo_update_with_anchor,
    static_argnames=(
        "optimizer",
        "accumulate_minibatches",
        "anchor_target_ratio",
        "anchor_epsilon",
        "anchor_maximum_scale",
    ),
)
