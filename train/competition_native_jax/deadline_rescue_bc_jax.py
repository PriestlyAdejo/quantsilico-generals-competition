"""Bounded Hybrid BC rescue with immutable original validation."""

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
from train.competition_native_jax.bc_warmstart_jax import sha256_file
from train.competition_native_jax.train_jax import load_tree, save_tree

NEG_LARGE = -1.0e30
ACTION_DIM = 3_970
TYPE_WEIGHT = 0.20
RANK_WEIGHT = 0.10
RANK_MARGIN = 0.50


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def action_type_targets(actions: jax.Array) -> jax.Array:
    local = jnp.mod(actions - 1, 9)
    return jnp.where(actions == 0, 0, jnp.where(local == 8, 2, 1)).astype(jnp.int32)


def action_type_masks() -> jax.Array:
    indices = jnp.arange(ACTION_DIM, dtype=jnp.int32)
    pass_mask = indices == 0
    build_mask = (indices > 0) & (jnp.mod(indices - 1, 9) == 8)
    move_mask = (indices > 0) & (~build_mask)
    return jnp.stack([pass_mask, move_mask, build_mask])


def action_type_logits(logits: jax.Array, legal_mask: jax.Array) -> jax.Array:
    type_masks = action_type_masks()

    def one_type(type_mask):
        valid = legal_mask & type_mask[None, :]
        masked = jnp.where(valid, logits, NEG_LARGE)
        return jax.nn.logsumexp(masked, axis=-1)

    return jax.vmap(one_type)(type_masks).T


def ranking_loss_per_sample(
    logits: jax.Array,
    legal_mask: jax.Array,
    actions: jax.Array,
    *,
    margin: float = RANK_MARGIN,
) -> jax.Array:
    teacher = jnp.take_along_axis(logits, actions[:, None], axis=1)[:, 0]
    wrong = legal_mask.at[jnp.arange(actions.shape[0]), actions].set(False)
    strongest_wrong = jnp.max(jnp.where(wrong, logits, NEG_LARGE), axis=1)
    return jnp.maximum(0.0, margin - teacher + strongest_wrong)


def hybrid_loss_and_metrics(
    params: dict,
    spatial: jax.Array,
    global_vec: jax.Array,
    legal_mask: jax.Array,
    actions: jax.Array,
    sample_weight: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    output = forward_batch(params, spatial, global_vec)
    logits = output["flat_logits"]
    logp = jax.vmap(masked_log_softmax)(logits, legal_mask)
    selected = jnp.take_along_axis(logp, actions[:, None], axis=1)[:, 0]
    exact_ce = -selected
    targets = action_type_targets(actions)
    type_logits = action_type_logits(logits, legal_mask)
    type_logp = jax.nn.log_softmax(type_logits, axis=-1)
    type_ce = -jnp.take_along_axis(type_logp, targets[:, None], axis=1)[:, 0]
    ranking = ranking_loss_per_sample(logits, legal_mask, actions)
    per_sample = exact_ce + TYPE_WEIGHT * type_ce + RANK_WEIGHT * ranking
    denominator = jnp.maximum(jnp.sum(sample_weight), 1.0)
    loss = jnp.sum(sample_weight * per_sample) / denominator
    prediction = jnp.argmax(jnp.where(legal_mask, logp, NEG_LARGE), axis=1)
    predicted_type = action_type_targets(prediction)
    legal_target = jnp.take_along_axis(legal_mask, actions[:, None], axis=1)[:, 0]
    teacher_rank = 1 + jnp.sum(
        legal_mask & (logits > jnp.take_along_axis(logits, actions[:, None], axis=1)),
        axis=1,
    )
    return loss, {
        "exact_ce": jnp.mean(exact_ce),
        "action_type_ce": jnp.mean(type_ce),
        "ranking_loss": jnp.mean(ranking),
        "top1": jnp.mean(prediction == actions),
        "top5": jnp.mean(teacher_rank <= 5),
        "mean_teacher_rank": jnp.mean(teacher_rank.astype(jnp.float32)),
        "action_type_accuracy": jnp.mean(predicted_type == targets),
        "legal_target_fraction": jnp.mean(legal_target),
        "finite_fraction": jnp.mean(jnp.isfinite(per_sample)),
    }


def zero_value_gradient(gradients: dict) -> dict:
    return {**gradients, "value_head": jnp.zeros_like(gradients["value_head"])}


def load_verified(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = json.loads(path.with_name("dataset_manifest.json").read_text(encoding="utf-8"))
    actual = sha256_file(path)
    if actual != manifest.get("dataset_sha256"):
        raise RuntimeError(f"dataset SHA mismatch: {path}")
    raw = np.load(path, allow_pickle=False)
    return manifest, {name: raw[name] for name in raw.files}


def validation_identity(data: dict[str, np.ndarray], indices: np.ndarray) -> str:
    digest = hashlib.sha256()
    for identity in sorted(data["sample_id"][indices].astype(str)):
        digest.update(identity.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def evaluate(
    params: dict,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, Any]:
    totals: dict[str, float] = {}
    phase_hits: dict[str, list[int]] = {}
    count = 0
    for start in range(0, len(indices), batch_size):
        selected_indices = indices[start : start + batch_size]
        weights = data.get("sample_weight", np.ones(len(data["teacher_action"]), dtype=np.float32))
        loss, metrics = hybrid_loss_and_metrics(
            params,
            jnp.asarray(data["spatial"][selected_indices], dtype=jnp.float32),
            jnp.asarray(data["global_vec"][selected_indices], dtype=jnp.float32),
            jnp.asarray(data["legal_mask"][selected_indices], dtype=bool),
            jnp.asarray(data["teacher_action"][selected_indices], dtype=jnp.int32),
            jnp.asarray(weights[selected_indices], dtype=jnp.float32),
        )
        loss = jax.block_until_ready(loss)
        size = len(selected_indices)
        totals["loss"] = totals.get("loss", 0.0) + float(loss) * size
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * size
        output = forward_batch(
            params,
            jnp.asarray(data["spatial"][selected_indices], dtype=jnp.float32),
            jnp.asarray(data["global_vec"][selected_indices], dtype=jnp.float32),
        )
        predictions = np.asarray(
            jnp.argmax(
                jnp.where(
                    jnp.asarray(data["legal_mask"][selected_indices], dtype=bool),
                    output["flat_logits"],
                    NEG_LARGE,
                ),
                axis=1,
            )
        )
        phases = (
            data["phase"][selected_indices].astype(str)
            if "phase" in data
            else np.where(
                data["turn"][selected_indices] < 128,
                "opening",
                np.where(data["turn"][selected_indices] < 512, "midgame", "conversion"),
            )
        )
        targets = data["teacher_action"][selected_indices]
        for phase in np.unique(phases):
            mask = phases == phase
            hit, total = phase_hits.setdefault(str(phase), [0, 0])
            phase_hits[str(phase)] = [
                hit + int(np.count_nonzero(predictions[mask] == targets[mask])),
                total + int(np.count_nonzero(mask)),
            ]
        count += size
    result = {key: value / max(count, 1) for key, value in totals.items()}
    result["samples"] = count
    result["phase_top1"] = {
        phase: hits / max(total, 1) for phase, (hits, total) in phase_hits.items()
    }
    return result


def stratified_batch_indices(
    rng: np.random.Generator,
    original_train: np.ndarray,
    dagger_disagreement: np.ndarray,
    dagger_recovery: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    original_count = batch_size // 2
    disagreement_count = int(round(batch_size * 0.30))
    recovery_count = batch_size - original_count - disagreement_count
    if not len(dagger_recovery):
        disagreement_count += recovery_count
        recovery_count = 0
    if not len(dagger_disagreement):
        recovery_count += disagreement_count
        disagreement_count = 0
    if not len(dagger_disagreement) and not len(dagger_recovery):
        raise RuntimeError("targeted DAgger training pool is empty")
    original = rng.choice(
        original_train,
        original_count,
        replace=len(original_train) < original_count,
    )
    targeted_parts: list[np.ndarray] = []
    targeted_sources: list[np.ndarray] = []
    if disagreement_count:
        targeted_parts.append(
            rng.choice(
                dagger_disagreement,
                disagreement_count,
                replace=len(dagger_disagreement) < disagreement_count,
            )
        )
        targeted_sources.append(np.ones(disagreement_count, dtype=np.int8))
    if recovery_count:
        targeted_parts.append(
            rng.choice(
                dagger_recovery,
                recovery_count,
                replace=len(dagger_recovery) < recovery_count,
            )
        )
        targeted_sources.append(np.full(recovery_count, 2, dtype=np.int8))
    targeted = np.concatenate(targeted_parts)
    source = np.concatenate([np.zeros(original_count, dtype=np.int8), *targeted_sources])
    return np.concatenate([original, targeted]), source


def batch_arrays(
    original: dict[str, np.ndarray],
    dagger: dict[str, np.ndarray],
    indices: np.ndarray,
    source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays: list[np.ndarray] = []
    for name, dtype in (
        ("spatial", np.float32),
        ("global_vec", np.float32),
        ("legal_mask", bool),
        ("teacher_action", np.int32),
    ):
        original_values = original[name][indices[source == 0]].astype(dtype, copy=False)
        targeted_values = dagger[name][indices[source != 0]].astype(dtype, copy=False)
        arrays.append(np.concatenate([original_values, targeted_values], axis=0))
    original_weight = np.ones(np.count_nonzero(source == 0), dtype=np.float32)
    targeted_weight = dagger["sample_weight"][indices[source != 0]].astype(np.float32)
    arrays.append(np.concatenate([original_weight, targeted_weight]))
    return tuple(arrays)  # type: ignore[return-value]


def train_rescue(
    *,
    original_dataset: Path,
    dagger_dataset: Path,
    parent_checkpoint: Path,
    output: Path,
    steps: int = 300,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    seed: int = 101,
    evaluation_interval: int = 50,
) -> dict[str, Any]:
    original_manifest, original = load_verified(original_dataset)
    dagger_manifest, dagger = load_verified(dagger_dataset)
    original_train = np.flatnonzero(original["split"] == "train")
    original_validation = np.flatnonzero(original["split"] == "validation")
    dagger_train = np.flatnonzero(dagger["split"] == "train")
    dagger_holdout = np.flatnonzero(dagger["split"] == "dagger_holdout")
    sources = dagger["sample_source"].astype(str)
    disagreement = dagger_train[sources[dagger_train] != "takeover_recovery"]
    recovery = dagger_train[sources[dagger_train] == "takeover_recovery"]
    if not len(original_train) or not len(original_validation) or not len(dagger_holdout):
        raise RuntimeError("immutable original validation / DAgger split gate failed")
    validation_hash_before = validation_identity(original, original_validation)

    params = load_tree(parent_checkpoint, init_params(jax.random.PRNGKey(0)))
    params = jax.device_put(params)
    parent_value_head = np.asarray(params["value_head"])
    baseline_original = evaluate(
        params, original, original_validation, batch_size=batch_size
    )
    baseline_dagger = evaluate(params, dagger, dagger_holdout, batch_size=batch_size)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(current, state, spatial, global_vec, legal_mask, actions, weights):
        def objective(candidate):
            return hybrid_loss_and_metrics(
                candidate, spatial, global_vec, legal_mask, actions, weights
            )

        (loss, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(current)
        gradients = zero_value_gradient(gradients)
        updates, state = optimizer.update(gradients, state, current)
        return optax.apply_updates(current, updates), state, loss, metrics

    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    history: list[dict[str, float]] = []
    validation_history: list[dict[str, Any]] = []
    best_params = params
    best_score = 0.0
    selected_step = 0
    for step in range(steps):
        combined_indices, source = stratified_batch_indices(
            rng,
            original_train,
            disagreement,
            recovery,
            batch_size=batch_size,
        )
        spatial, global_vec, legal_mask, actions, weights = batch_arrays(
            original, dagger, combined_indices, source
        )
        params, opt_state, loss, metrics = train_step(
            params,
            opt_state,
            jnp.asarray(spatial),
            jnp.asarray(global_vec),
            jnp.asarray(legal_mask),
            jnp.asarray(actions),
            jnp.asarray(weights),
        )
        loss = jax.block_until_ready(loss)
        if step == 0 or (step + 1) % 25 == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss),
                    **{key: float(value) for key, value in metrics.items()},
                }
            )
        if (step + 1) % evaluation_interval == 0:
            original_metrics = evaluate(
                params, original, original_validation, batch_size=batch_size
            )
            dagger_metrics = evaluate(params, dagger, dagger_holdout, batch_size=batch_size)
            ce_gain = (baseline_original["exact_ce"] - original_metrics["exact_ce"]) / max(
                baseline_original["exact_ce"], 1e-9
            )
            top1_gain = original_metrics["top1"] - baseline_original["top1"]
            score = ce_gain + top1_gain
            validation_history.append(
                {
                    "step": step + 1,
                    "ORIGINAL_FIXED_VALIDATION_METRICS": original_metrics,
                    "DAGGER_HOLDOUT_METRICS": dagger_metrics,
                    "selection_score": score,
                }
            )
            if score > best_score:
                best_score = score
                best_params = params
                selected_step = step + 1
    params = best_params
    final_original = evaluate(params, original, original_validation, batch_size=batch_size)
    final_dagger = evaluate(params, dagger, dagger_holdout, batch_size=batch_size)
    validation_hash_after = validation_identity(original, original_validation)
    if validation_hash_after != validation_hash_before:
        raise RuntimeError("ORIGINAL_FIXED_VALIDATION mutation detected")
    if not np.array_equal(np.asarray(params["value_head"]), parent_value_head):
        raise RuntimeError("value head changed during Hybrid BC")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "rescue_raw.npz"
    save_tree(checkpoint, params)
    report = {
        "schema_version": 1,
        "kind": "NOON_DEADLINE_HYBRID_BC",
        "status": "DEADLINE_HYBRID_BC_TRAINED",
        "gameplay_gate_required_next": True,
        "original_dataset": str(original_dataset),
        "original_dataset_sha256": original_manifest["dataset_sha256"],
        "dagger_dataset": str(dagger_dataset),
        "dagger_dataset_sha256": dagger_manifest["dataset_sha256"],
        "parent_checkpoint": str(parent_checkpoint),
        "parent_checkpoint_sha256": sha256_file(parent_checkpoint),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "value_head_frozen": True,
        "fresh_bc_optimizer": True,
        "loss": {
            "exact_legal_ce": 1.0,
            "action_type_ce": TYPE_WEIGHT,
            "strongest_wrong_legal_ranking": RANK_WEIGHT,
            "ranking_margin": RANK_MARGIN,
        },
        "sampling": {
            "original": 0.50,
            "dagger_disagreement": 0.30,
            "takeover_recovery": 0.20,
        },
        "steps": steps,
        "selected_step": selected_step,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "ORIGINAL_FIXED_VALIDATION_IDENTITY": validation_hash_before,
        "ORIGINAL_FIXED_VALIDATION_METRICS": {
            "baseline": baseline_original,
            "final": final_original,
        },
        "DAGGER_HOLDOUT_METRICS": {
            "baseline": baseline_dagger,
            "final": final_dagger,
        },
        "CLOSED_LOOP_GAMEPLAY_METRICS": "PENDING_MANDATORY_GATE",
        "history": history,
        "validation_history": validation_history,
        "device": [str(device) for device in jax.devices()],
        "elapsed_s": time.perf_counter() - started,
        "written_at": datetime.now(UTC).isoformat(),
    }
    atomic_json(output / "hybrid_bc_report.json", report)
    return report
