"""STAGE5_TEACHER_R1 STEP2+STEP3: label extraction + BC distillation.

STEP2: replays each STEP1 transcript through the pinned engine and pairs the
hunter-seat canonical legal observation (observe_one_jax with tracked fog
memory - same legal path as BC-A, EV-0050/0060) with the action the teacher
actually executed that tick (engine-verified by construction). Labels whose
action index falls outside legal_mask_one_jax are ENGINE_SILENT_PASS and are
excluded (predeclared). Only hunter-WON games contribute labels.

STEP3: trains the BC-A canonical small CNN (same architecture/hyperparameters
as bc_a_train_pilot.py) on the teacher data with game-disjoint splits
(train 14 / holdout_games 3 / holdout_maps 3). Screening gate is IDENTICAL
to BC-A-FULL (EV-0060): held-out top-1 strictly above BOTH legal-uniform and
majority-pass on >= 1 holdout split. PPO_SEMANTICS: OFF_POLICY_AUXILIARY.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src", REPO / "third_party" / "generals-bots"):
    entry = str(candidate)
    if entry not in sys.path:
        sys.path.insert(0, entry)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from generals import GeneralsEnv  # noqa: E402

from generals_bot.competition_native_jax.competition_env_jax import (  # noqa: E402
    empty_memory,
    legal_mask_one_jax,
    observe_one_jax,
)
from generals_bot.evaluation.match import make_board, make_transition  # noqa: E402
from scripts.training.bc_a_train_pilot import (  # noqa: E402
    ACTION_DIM,
    PASS_INDEX,
    init_params,
    masked_accuracy,
    small_policy,
)

STEP1_ROOT = REPO / "experiments/marathon/teacher_r1/step1_generation"
OUT_ROOT = REPO / "experiments/marathon/teacher_r1/step3_bc"
MAX_HW = 21


def engine_action_to_index(action) -> int:
    kind, row, col, direction, split = action
    if kind == 1:
        return 0
    cell = row * MAX_HW + col
    if kind == 2:
        return 1 + cell * 9 + 8
    return 1 + cell * 9 + direction * 2 + split


def build_features(docs: list[dict], split_of: dict[int, str]) -> dict[str, dict]:
    env = GeneralsEnv(mode="competition")
    transition = make_transition(env)
    features: dict[str, dict] = {
        split: {"spatial": [], "global": [], "mask": [], "label": []}
        for split in ("train", "holdout_games", "holdout_maps")
    }
    silent_pass_excluded = 0
    for doc in docs:
        seed = doc["seed"]
        split = split_of[seed]
        state = make_board(env, seed)
        memory = empty_memory()
        for a0, a1 in zip(doc["actions_hunter"], doc["actions_opponent"]):
            spatial, global_vec, memory = observe_one_jax(state, 0, memory)
            mask = legal_mask_one_jax(state, 0)
            idx = engine_action_to_index(a0)
            if bool(mask[idx]):
                bucket = features[split]
                bucket["spatial"].append(np.asarray(spatial, dtype=np.float16))
                bucket["global"].append(np.asarray(global_vec, dtype=np.float16))
                bucket["mask"].append(np.asarray(mask, dtype=bool))
                bucket["label"].append(idx)
            else:
                silent_pass_excluded += 1
            state, _info = transition(
                state, jnp.stack([jnp.array(a0, dtype=jnp.int32),
                                  jnp.array(a1, dtype=jnp.int32)])
            )
        print(f"features: seed={seed} split={split} done", flush=True)
    return {
        "features": {
            split: {k: (np.stack(v) if v else np.zeros((0,))) for k, v in arrays.items()}
            for split, arrays in features.items()
        },
        "silent_pass_excluded": silent_pass_excluded,
    }


def evaluate(params, split: dict, name: str) -> dict:
    spatial = jnp.asarray(split["spatial"], dtype=jnp.float32)
    global_vec = jnp.asarray(split["global"], dtype=jnp.float32)
    masks = jnp.asarray(split["mask"])
    labels = jnp.asarray(split["label"])
    logits = small_policy(params, spatial, global_vec)
    acc, legal_share = masked_accuracy(logits, masks, labels)
    legal_counts = np.asarray(masks).sum(axis=1)
    labels_np = np.asarray(labels)
    return {
        "top1_accuracy": round(float(acc), 6),
        "legal_action_share": round(float(legal_share), 6),
        "legal_uniform_baseline": round(float(np.mean(1.0 / np.maximum(legal_counts, 1))), 6),
        "majority_pass_baseline": round(float(np.mean(labels_np == PASS_INDEX)), 6),
        "n": int(labels_np.shape[0]),
        "split": name,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()

    summary = json.loads((STEP1_ROOT / "summary.json").read_text(encoding="utf-8"))
    if not summary["gate_pass"]:
        print("STEP1 GATE NOT PASSED - refusing to extract labels", flush=True)
        return 3
    docs = json.loads((STEP1_ROOT / "transcripts.json").read_text(encoding="utf-8"))
    won = [d for d in docs if d["winner"] == 0 and not d["truncated"]]
    seeds = sorted(d["seed"] for d in won)
    split_of = {
        **{s: "train" for s in seeds[:14]},
        **{s: "holdout_games" for s in seeds[14:17]},
        **{s: "holdout_maps" for s in seeds[17:20]},
    }
    print(f"won games: {len(won)}/20; splits train={len(seeds[:14])} "
          f"holdout_games={len(seeds[14:17])} holdout_maps={len(seeds[17:20])}", flush=True)

    built = build_features(won, split_of)
    features = built["features"]
    for split, arrays in features.items():
        print(f"split {split}: n={arrays['label'].shape[0]}", flush=True)

    train = features["train"]
    spatial = jnp.asarray(train["spatial"], dtype=jnp.float32)
    global_vec = jnp.asarray(train["global"], dtype=jnp.float32)
    labels = jnp.asarray(train["label"])
    n = int(labels.shape[0])

    key = jax.random.PRNGKey(args.seed)
    params = init_params(jax.random.fold_in(key, 1))
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, x_s, x_g, masks, y):
        logits = small_policy(params, x_s, x_g)
        masked = jnp.where(masks.astype(bool), logits, jnp.finfo(jnp.float32).min)
        logp = jax.nn.log_softmax(masked, axis=1)
        return -jnp.mean(jnp.take_along_axis(logp, y[:, None], axis=1))

    @jax.jit
    def step(params, opt_st, x_s, x_g, masks, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x_s, x_g, masks, y)
        updates, opt_st = optimizer.update(grads, opt_st, params)
        return optax.apply_updates(params, updates), opt_st, loss

    train_masks = jnp.asarray(train["mask"])
    rng = np.random.default_rng(args.seed)
    history = []
    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        order = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, args.batch_size):
            idx = order[start:start + args.batch_size]
            params, opt_state, loss = step(
                params, opt_state, spatial[idx], global_vec[idx],
                train_masks[idx], labels[idx]
            )
            epoch_loss += float(loss)
        history.append(epoch_loss / max(1, n // args.batch_size))
        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch + 1}/{args.epochs} loss={history[-1]:.4f}", flush=True)

    results = {
        "kind": "STAGE5_TEACHER_R1_BC_RESULT",
        "plan": "experiments/marathon/stage5_teacher_r1_plan.yaml",
        "experiment_id": "experiment#stage5-teacher-r1#153b464617b7",
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "step1_summary_sha_seedcount": len(docs),
        "won_games": len(won),
        "won_seeds": seeds,
        "splits": {"train": seeds[:14], "holdout_games": seeds[14:17],
                   "holdout_maps": seeds[17:20]},
        "silent_pass_excluded": built["silent_pass_excluded"],
        "train_seconds": round(time.perf_counter() - t_start, 1),
        "final_loss": history[-1],
        "train": evaluate(params, train, "train"),
        "holdout_games": evaluate(params, features["holdout_games"], "holdout_games"),
        "holdout_maps": evaluate(params, features["holdout_maps"], "holdout_maps"),
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    def gate_pass(block: dict) -> bool:
        return (block["top1_accuracy"] > block["legal_uniform_baseline"]
                and block["top1_accuracy"] > block["majority_pass_baseline"])

    results["screening_gate"] = {
        "rule": "top1 strictly above BOTH legal_uniform AND majority_pass on >=1 holdout split (EV-0060-identical)",
        "holdout_games_pass": gate_pass(results["holdout_games"]),
        "holdout_maps_pass": gate_pass(results["holdout_maps"]),
    }
    results["screening_gate"]["verdict"] = (
        "PASS_GATE" if (results["screening_gate"]["holdout_games_pass"]
                        or results["screening_gate"]["holdout_maps_pass"])
        else "VALID_NEGATIVE"
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    result_path = OUT_ROOT / "teacher_r1_result.json"
    result_path.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8", newline="\n")
    arrays = {}
    for k, v in params.items():
        if isinstance(v, list):
            for i, arr in enumerate(v):
                arrays[f"{k}_{i}"] = np.asarray(arr)
        else:
            arrays[k] = np.asarray(v)
    np.savez(OUT_ROOT / "params.npz", **arrays)
    print(json.dumps(results, indent=1), flush=True)
    print(results["screening_gate"]["verdict"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
