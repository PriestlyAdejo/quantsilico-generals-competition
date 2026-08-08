"""Post-BC value-head recalibration with the warm-started policy frozen."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from generals_bot.competition_native_jax.transformer_jax import forward_batch, init_params
from train.competition_native_jax.bc_warmstart_jax import sha256_file
from train.competition_native_jax.gae_jax import gae_advantages_batch_jit
from train.competition_native_jax.opponents_jax import build_static_schedule
from train.competition_native_jax.ppo_jax import hl_gauss_target
from train.competition_native_jax.rollout_curriculum_jax import collect_curriculum_batch
from train.competition_native_jax.train_jax import load_tree, save_tree


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def calibrate_value_head(
    *,
    warmstart_checkpoint: Path,
    output: Path,
    num_envs: int = 32,
    rollout_len: int = 1_200,
    steps: int = 200,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    gamma: float = 1.0,
    lam: float = 0.9,
    seed: int = 53,
) -> dict[str, Any]:
    params = load_tree(warmstart_checkpoint, init_params(jax.random.PRNGKey(0)))
    params = jax.device_put(params)
    schedule = build_static_schedule(num_envs, (0.4, 0.4, 0.2, 0.0))
    started = time.perf_counter()
    batch, _carry = collect_curriculum_batch(
        params,
        opponent_schedule=schedule,
        rollout_len=rollout_len,
        seed=seed,
        gamma=gamma,
        shaping_lambda=0.0,
    )
    values = jnp.concatenate([batch["values"], batch["bootstrap_values"][None, :]], axis=0)
    _advantages, returns = gae_advantages_batch_jit(
        batch["rewards"], values, batch["dones"], gamma=gamma, lam=lam
    )
    flat_spatial = batch["spatial"].reshape(
        (-1,) + batch["spatial"].shape[2:]
    )
    flat_global = batch["global"].reshape((-1,) + batch["global"].shape[2:])
    flat_returns = returns.reshape(-1)
    if not bool(jnp.all(jnp.isfinite(flat_returns))):
        raise RuntimeError("nonfinite post-BC value targets")
    completed = int(jnp.sum(batch["dones"]))
    nonzero_terminal = int(jnp.count_nonzero(batch["terminal_rewards"]))
    if completed < 32 or nonzero_terminal < 16:
        raise RuntimeError(
            "value calibration trajectories lack terminal task signal: "
            f"completed={completed} nonzero_terminal={nonzero_terminal}"
        )

    feature_params = {key: value for key, value in params.items() if key != "value_head"}
    value_head = params["value_head"]
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(value_head)

    def metrics_for(head) -> dict[str, float]:
        losses: list[float] = []
        predicted: list[np.ndarray] = []
        target_values: list[np.ndarray] = []
        for start in range(0, flat_returns.shape[0], batch_size):
            stop = min(start + batch_size, flat_returns.shape[0])
            candidate = {**feature_params, "value_head": head}
            output_value = forward_batch(
                candidate, flat_spatial[start:stop], flat_global[start:stop]
            )["value_logits"]
            target = jax.vmap(hl_gauss_target)(flat_returns[start:stop])
            loss = -jnp.mean(
                jnp.sum(target * jax.nn.log_softmax(output_value, axis=-1), axis=-1)
            )
            centers = jnp.linspace(-1.0, 1.0, output_value.shape[-1])
            prediction = jnp.sum(
                jax.nn.softmax(output_value, axis=-1) * centers, axis=-1
            )
            losses.append(float(loss) * (stop - start))
            predicted.append(np.asarray(prediction))
            target_values.append(np.asarray(flat_returns[start:stop]))
        pred = np.concatenate(predicted)
        target = np.concatenate(target_values)
        target_variance = float(np.var(target))
        explained = (
            1.0 - float(np.var(target - pred)) / target_variance
            if target_variance > 1e-12
            else 0.0
        )
        return {
            "loss": sum(losses) / len(target),
            "prediction_variance": float(np.var(pred)),
            "target_variance": target_variance,
            "explained_variance": explained,
            "finite_prediction_fraction": float(np.mean(np.isfinite(pred))),
        }

    baseline = metrics_for(value_head)
    rng = np.random.default_rng(seed)

    @jax.jit
    def step(head, state, spatial, global_vec, targets):
        def objective(candidate):
            full = {**feature_params, "value_head": candidate}
            logits = forward_batch(full, spatial, global_vec)["value_logits"]
            distribution = jax.vmap(hl_gauss_target)(targets)
            return -jnp.mean(
                jnp.sum(distribution * jax.nn.log_softmax(logits, axis=-1), axis=-1)
            )

        loss, gradient = jax.value_and_grad(objective)(head)
        updates, state = optimizer.update(gradient, state, head)
        return optax.apply_updates(head, updates), state, loss

    history: list[dict[str, float]] = []
    count = int(flat_returns.shape[0])
    for index in range(steps):
        selected = rng.integers(0, count, size=batch_size)
        value_head, opt_state, loss = step(
            value_head,
            opt_state,
            flat_spatial[selected],
            flat_global[selected],
            flat_returns[selected],
        )
        if index == 0 or (index + 1) % 25 == 0:
            history.append({"step": index + 1, "loss": float(loss)})

    final = metrics_for(value_head)
    loss_improvement = (baseline["loss"] - final["loss"]) / max(baseline["loss"], 1e-9)
    passed = bool(
        final["finite_prediction_fraction"] == 1.0
        and final["target_variance"] > 1e-6
        and final["prediction_variance"] > 1e-6
        and final["explained_variance"] > -0.25
        and loss_improvement >= 0.20
    )
    calibrated = {**feature_params, "value_head": value_head}
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "post_bc_value_calibrated_raw.npz"
    save_tree(checkpoint, calibrated)
    report = {
        "schema_version": 1,
        "kind": "POST_BC_VALUE_CALIBRATION",
        "status": "POST_BC_VALUE_CALIBRATION_PASS" if passed else "POST_BC_VALUE_CALIBRATION_FAIL",
        "warmstart_checkpoint": str(warmstart_checkpoint),
        "warmstart_sha256": sha256_file(warmstart_checkpoint),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "policy_frozen": True,
        "fresh_value_optimizer": True,
        "fresh_ppo_optimizer_required_next": True,
        "trajectory": {
            "num_envs": num_envs,
            "rollout_len": rollout_len,
            "schedule_counts": [schedule.count(kind) for kind in range(4)],
            "completed_episodes": completed,
            "nonzero_terminal_rewards": nonzero_terminal,
            "gamma": gamma,
            "lambda": lam,
        },
        "baseline": baseline,
        "final": final,
        "value_loss_relative_improvement": loss_improvement,
        "history": history,
        "elapsed_s": time.perf_counter() - started,
        "written_at": datetime.now(UTC).isoformat(),
    }
    atomic_json(output / "post_bc_value_calibration_report.json", report)
    return report
