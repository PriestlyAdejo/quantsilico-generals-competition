"""STAGE6_DISTILL_PPO_R1 STEP D0: distill TEACHER-R2 teacher data into the
canonical transformer (predeclared: stage6_distill_ppo_r1_plan.yaml).

Feature path is IDENTICAL to TEACHER-R2 STEP2 (both-seat engine-replay labels
through the canonical legal observation path; silent-pass excluded; same
deterministic splits). The model is the canonical transformer_jax policy
(8-plane legal observation -> 3970 action logits) trained with the same
mask-restricted log-softmax cross-entropy as the BC lineage.

Output is a warm-start checkpoint directory (raw.npz / ema.npz /
opt_state.npz / meta.json) consumable by run_sh_r1_arm.py --checkpoint for
the D1 PPO continuation arms. EMA is the BC parameters themselves (no EMA
during distillation - declared). Screening gate is EV-0060-identical.
PPO_SEMANTICS: OFF_POLICY_AUXILIARY (distillation data never enters
on-policy PPO; the checkpoint is an initialization only).
"""

from __future__ import annotations

import argparse
import hashlib
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

from generals_bot.competition_native_jax.transformer_jax import (  # noqa: E402
    forward,
    init_params,
)
from scripts.training.bc_a_train_pilot import PASS_INDEX, masked_accuracy  # noqa: E402
from scripts.training.teacher_r2_train import build_features  # noqa: E402
from train.competition_native_jax.ppo_jax import make_optimizer  # noqa: E402
from train.competition_native_jax.train_jax import save_training_checkpoint  # noqa: E402

STEP1_ROOT = REPO / "experiments/marathon/teacher_r2/step1_selfplay"
OUT_ROOT = REPO / "experiments/marathon/distill_ppo_r1/d0_distill/DISTILL-S0-TRANSFORMER-BC"


def evaluate(params, split: dict, name: str) -> dict:
    spatial = jnp.asarray(split["spatial"], dtype=jnp.float32)
    global_vec = jnp.asarray(split["global"], dtype=jnp.float32)
    masks = jnp.asarray(split["mask"])
    labels = jnp.asarray(split["label"])
    logits = jax.vmap(forward, in_axes=(None, 0, 0))(params, spatial, global_vec)["flat_logits"]
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--smoke-games", type=int, default=0,
                        help="if >0, restrict to this many decisive games (CPU smoke)")
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    summary = json.loads((STEP1_ROOT / "summary.json").read_text(encoding="utf-8"))
    if not summary["gate_pass"]:
        print("TEACHER-R2 STEP1 GATE NOT PASSED - refusing to distill", flush=True)
        return 3
    docs = json.loads((STEP1_ROOT / "transcripts.json").read_text(encoding="utf-8"))
    decisive = [d for d in docs if d["winner"] in (0, 1) and not d["truncated"]]
    seeds = sorted(d["seed"] for d in decisive)
    split_of = {
        **{s: "train" for s in seeds[:14]},
        **{s: "holdout_games" for s in seeds[14:17]},
        **{s: "holdout_maps" for s in seeds[17:20]},
    }
    if args.smoke_games > 0:
        keep = set(seeds[:args.smoke_games])
        decisive = [d for d in decisive if d["seed"] in keep]
        print(f"SMOKE: restricted to games {sorted(keep)}", flush=True)
    print(f"distilling from {len(decisive)} decisive games "
          f"(train={len([s for s in seeds[:14] if any(d['seed'] == s for d in decisive)])})",
          flush=True)

    built = build_features(decisive, split_of)
    features = built["features"]
    for split, arrays in features.items():
        print(f"split {split}: n={arrays['label'].shape[0]}", flush=True)

    train = features["train"]
    spatial = jnp.asarray(train["spatial"], dtype=jnp.float32)
    global_vec = jnp.asarray(train["global"], dtype=jnp.float32)
    train_masks = jnp.asarray(train["mask"])
    labels = jnp.asarray(train["label"])
    n = int(labels.shape[0])

    key = jax.random.PRNGKey(args.seed)
    params = init_params(jax.random.fold_in(key, 1))
    optimizer = make_optimizer(args.lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, x_s, x_g, masks, y):
        logits = jax.vmap(forward, in_axes=(None, 0, 0))(params, x_s, x_g)["flat_logits"]
        masked = jnp.where(masks.astype(bool), logits, jnp.finfo(jnp.float32).min)
        logp = jax.nn.log_softmax(masked, axis=1)
        return -jnp.mean(jnp.take_along_axis(logp, y[:, None], axis=1))

    @jax.jit
    def step(params, opt_st, x_s, x_g, masks, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x_s, x_g, masks, y)
        updates, opt_st = optimizer.update(grads, opt_st, params)
        return optax.apply_updates(params, updates), opt_st, loss

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
        print(f"epoch {epoch + 1}/{args.epochs} loss={history[-1]:.4f}", flush=True)

    results = {
        "kind": "STAGE6_DISTILL_PPO_R1_D0_RESULT",
        "plan": "experiments/marathon/stage6_distill_ppo_r1_plan.yaml",
        "experiment_id": "experiment#stage6-distill-ppo-r1#543924456945",
        "model": "canonical transformer_jax (8-plane legal obs, 3970 action head)",
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "smoke_games": args.smoke_games,
        "decisive_games": len(decisive),
        "splits": {"train": seeds[:14], "holdout_games": seeds[14:17],
                   "holdout_maps": seeds[17:20]},
        "silent_pass_excluded": built["silent_pass_excluded"],
        "train_seconds": round(time.perf_counter() - t_start, 1),
        "final_loss": history[-1],
        "train": evaluate(params, train, "train"),
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for split_name in ("holdout_games", "holdout_maps"):
        if features[split_name]["label"].shape[0] > 0:
            results[split_name] = evaluate(params, features[split_name], split_name)
        else:
            results[split_name] = {"split": split_name, "n": 0,
                                   "note": "EMPTY_SPLIT_no_deterministic_seeds_assigned"}

    def gate_pass(block: dict) -> bool:
        if block.get("n", 0) == 0:
            return False
        return (block["top1_accuracy"] > block["legal_uniform_baseline"]
                and block["top1_accuracy"] > block["majority_pass_baseline"]
                and block["legal_action_share"] >= 0.999)

    results["screening_gate"] = {
        "rule": "top1 strictly above BOTH legal_uniform AND majority_pass AND legal_share 1.0 on >=1 holdout split (EV-0060-identical)",
        "holdout_games_pass": gate_pass(results["holdout_games"]),
        "holdout_maps_pass": gate_pass(results["holdout_maps"]),
        "smoke": args.smoke_games > 0,
    }
    results["screening_gate"]["verdict"] = (
        "SMOKE_ONLY" if args.smoke_games > 0 else
        ("PASS_GATE" if (results["screening_gate"]["holdout_games_pass"]
                         or results["screening_gate"]["holdout_maps_pass"])
         else "VALID_NEGATIVE")
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out.parent / "distill_d0_result.json").write_text(
        json.dumps(results, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    if args.smoke_games == 0 and results["screening_gate"]["verdict"] == "PASS_GATE":
        save_training_checkpoint(
            out, params=params, ema=params, opt_state=opt_state,
            meta={
                "kind": "STAGE6_DISTILL_PPO_R1_D0_CHECKPOINT",
                "source_dataset_transcript_sha256": summary["transcript_sha256"],
                "teacher_r2_result": "experiments/marathon/teacher_r2/step3_bc/teacher_r2_result.json",
                "ema_note": "EMA = BC params (no EMA during distillation, declared)",
                "ppo_semantics": "OFF_POLICY_AUXILIARY initialization",
                "finished_at_utc": results["finished_at_utc"],
            },
        )
        results["checkpoint"] = {
            "dir": str(out),
            "raw_sha256": sha256_file(out / "raw.npz"),
            "ema_sha256": sha256_file(out / "ema.npz"),
        }
        (out.parent / "distill_d0_result.json").write_text(
            json.dumps(results, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
        print("CHECKPOINT_SAVED", flush=True)
    print(json.dumps(results, indent=1), flush=True)
    print(results["screening_gate"]["verdict"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
