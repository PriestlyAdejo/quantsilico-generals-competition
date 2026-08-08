"""Legal-masked behaviour cloning for the competition-native transformer."""

from __future__ import annotations

import hashlib
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

from generals_bot.competition_native_jax.inference_jax import masked_log_softmax
from generals_bot.competition_native_jax.transformer_jax import forward_batch, init_params
from train.competition_native_jax.train_jax import load_tree, save_tree


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def policy_loss_and_metrics(
    params: dict,
    spatial: jax.Array,
    global_vec: jax.Array,
    legal_mask: jax.Array,
    actions: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    output = forward_batch(params, spatial, global_vec)
    logp = jax.vmap(masked_log_softmax)(output["flat_logits"], legal_mask)
    selected = jnp.take_along_axis(logp, actions[:, None], axis=1)[:, 0]
    prediction = jnp.argmax(logp, axis=1)
    legal_target = jnp.take_along_axis(legal_mask, actions[:, None], axis=1)[:, 0]
    return -jnp.mean(selected), {
        "top1": jnp.mean(prediction == actions),
        "legal_target_fraction": jnp.mean(legal_target),
        "finite_fraction": jnp.mean(jnp.isfinite(selected)),
    }


def evaluate(
    params: dict,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, float]:
    weighted = {"loss": 0.0, "top1": 0.0, "legal_target_fraction": 0.0, "finite_fraction": 0.0}
    count = 0
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        loss, metrics = policy_loss_and_metrics(
            params,
            jnp.asarray(data["spatial"][idx], dtype=jnp.float32),
            jnp.asarray(data["global_vec"][idx], dtype=jnp.float32),
            jnp.asarray(data["legal_mask"][idx], dtype=bool),
            jnp.asarray(data["teacher_action"][idx], dtype=jnp.int32),
        )
        jax.block_until_ready(loss)
        size = len(idx)
        weighted["loss"] += float(loss) * size
        for key in metrics:
            weighted[key] += float(metrics[key]) * size
        count += size
    return {key: value / max(count, 1) for key, value in weighted.items()}


def train_warmstart(
    *,
    dataset_path: Path,
    parent_checkpoint: Path,
    output: Path,
    steps: int,
    batch_size: int,
    learning_rate: float,
    wall_minutes: float,
    seed: int,
    evaluation_interval: int = 50,
    trainable_scope: str = "full_policy",
) -> dict[str, Any]:
    manifest_path = dataset_path.with_name("dataset_manifest.json")
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_dataset_sha = sha256_file(dataset_path)
    if actual_dataset_sha != dataset_manifest["dataset_sha256"]:
        raise RuntimeError("BC dataset SHA mismatch")
    raw = np.load(dataset_path, allow_pickle=False)
    data = {name: raw[name] for name in raw.files}
    train_indices = np.flatnonzero(data["split"] == "train")
    validation_indices = np.flatnonzero(data["split"] == "validation")
    if not len(train_indices) or not len(validation_indices):
        raise RuntimeError("BC dataset must contain complete-game train and validation splits")

    template = init_params(jax.random.PRNGKey(0))
    params = load_tree(parent_checkpoint / "raw.npz", template)
    params = jax.device_put(params)
    baseline = evaluate(params, data, validation_indices, batch_size=batch_size)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(current, state, spatial, global_vec, legal_mask, actions):
        def objective(candidate):
            return policy_loss_and_metrics(
                candidate, spatial, global_vec, legal_mask, actions
            )

        (loss, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(current)
        # The u1524 critic is intentionally frozen throughout BC.  It is
        # recalibrated in a separate, mandatory post-BC phase.
        gradients = {**gradients, "value_head": jnp.zeros_like(gradients["value_head"])}
        if trainable_scope in {"policy_heads", "last_layer_heads"}:
            trainable = {"move_head", "build_head", "pass_head"}
            gradients = {
                key: value
                if key in trainable or (key == "layers" and trainable_scope == "last_layer_heads")
                else jax.tree_util.tree_map(jnp.zeros_like, value)
                for key, value in gradients.items()
            }
            if trainable_scope == "last_layer_heads":
                gradients["layers"] = [
                    jax.tree_util.tree_map(jnp.zeros_like, layer)
                    for layer in gradients["layers"][:-1]
                ] + [gradients["layers"][-1]]
        updates, state = optimizer.update(gradients, state, current)
        current = optax.apply_updates(current, updates)
        return current, state, loss, metrics

    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    history: list[dict[str, float]] = []
    validation_history: list[dict[str, float]] = []
    best_params = params
    best_validation = baseline
    best_score = 0.0
    selected_step = 0
    steps_ran = 0
    for step in range(steps):
        if time.perf_counter() - started >= wall_minutes * 60:
            break
        replace = len(train_indices) < batch_size
        idx = rng.choice(train_indices, size=batch_size, replace=replace)
        params, opt_state, loss, metrics = train_step(
            params,
            opt_state,
            jnp.asarray(data["spatial"][idx], dtype=jnp.float32),
            jnp.asarray(data["global_vec"][idx], dtype=jnp.float32),
            jnp.asarray(data["legal_mask"][idx], dtype=bool),
            jnp.asarray(data["teacher_action"][idx], dtype=jnp.int32),
        )
        jax.block_until_ready(loss)
        steps_ran = step + 1
        if step == 0 or (step + 1) % 25 == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss),
                    **{key: float(value) for key, value in metrics.items()},
                }
            )

        if (step + 1) % evaluation_interval == 0:
            candidate = evaluate(
                params, data, validation_indices, batch_size=batch_size
            )
            candidate_ce = (baseline["loss"] - candidate["loss"]) / max(
                baseline["loss"], 1e-9
            )
            candidate_top1 = candidate["top1"] - baseline["top1"]
            score = candidate_ce + candidate_top1
            validation_history.append(
                {
                    "step": step + 1,
                    **candidate,
                    "ce_relative_improvement": candidate_ce,
                    "top1_absolute_gain": candidate_top1,
                    "selection_score": score,
                }
            )
            if score > best_score:
                best_score = score
                best_params = params
                best_validation = candidate
                selected_step = step + 1
            if candidate_ce >= 0.20 and candidate_top1 >= 0.15:
                break

    if steps_ran % evaluation_interval:
        candidate = evaluate(params, data, validation_indices, batch_size=batch_size)
        candidate_ce = (baseline["loss"] - candidate["loss"]) / max(
            baseline["loss"], 1e-9
        )
        candidate_top1 = candidate["top1"] - baseline["top1"]
        score = candidate_ce + candidate_top1
        validation_history.append(
            {
                "step": steps_ran,
                **candidate,
                "ce_relative_improvement": candidate_ce,
                "top1_absolute_gain": candidate_top1,
                "selection_score": score,
            }
        )
        if score > best_score:
            best_params = params
            best_validation = candidate
            selected_step = steps_ran
    params = best_params
    final = best_validation
    ce_improvement = (baseline["loss"] - final["loss"]) / max(baseline["loss"], 1e-9)
    top1_gain = final["top1"] - baseline["top1"]
    numeric_pass = bool(
        final["legal_target_fraction"] == 1.0
        and final["finite_fraction"] == 1.0
        and ce_improvement >= 0.20
        and top1_gain >= 0.15
    )

    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "warmstart_raw.npz"
    save_tree(checkpoint, params)
    checkpoint_sha = sha256_file(checkpoint)
    report = {
        "schema_version": 1,
        "kind": "VALID_LEARNING_BC_WARMSTART",
        "status": "BC_NUMERIC_GATE_PASS" if numeric_pass else "BC_NUMERIC_GATE_FAIL",
        "dataset": str(dataset_path),
        "dataset_sha256": actual_dataset_sha,
        "parent_checkpoint": str(parent_checkpoint),
        "parent_raw_sha256": sha256_file(parent_checkpoint / "raw.npz"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "value_head_frozen": True,
        "trainable_scope": trainable_scope,
        "optimizer_reused_from_u1524": False,
        "steps_ran": steps_ran,
        "selected_step": selected_step,
        "selection": "best whole-game validation checkpoint; stop at first full numeric gate",
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "baseline_validation": baseline,
        "final_validation": final,
        "validation_ce_relative_improvement": ce_improvement,
        "validation_top1_absolute_gain": top1_gain,
        "required": {
            "ce_relative_improvement": 0.20,
            "top1_absolute_gain": 0.15,
            "legal_target_fraction": 1.0,
            "finite_fraction": 1.0,
        },
        "history": history,
        "validation_history": validation_history,
        "device": [str(device) for device in jax.devices()],
        "elapsed_s": time.perf_counter() - started,
        "written_at": datetime.now(UTC).isoformat(),
    }
    atomic_json(output / "bc_warmstart_report.json", report)
    return report
