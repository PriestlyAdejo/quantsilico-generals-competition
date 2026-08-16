"""BC-A pilot trainer (predeclared: bc_a_pilot_round_1_plan.yaml).

Trains a SMALL standalone policy (fresh init; consumes the canonical legal
observation tensors spatial[8,21,21] + global[8] produced by observe_one_jax)
on behavioural-cloning cross-entropy over the 3970-dim competition action
space, legal-masked. Seed 20260831 per predeclaration; CPU budget <= 30 min.

Features are reconstructed ONLY through the canonical legal observation path
(state_from_tick -> observe_one_jax with per-seat fog memory tracked
sequentially from tick 0); full hidden state never enters features.

Metrics (predeclared): TRAIN top-1 accuracy, HELD-OUT-PLAYER (ResBot) top-1,
HELD-OUT-REPLAY (time-disjoint) top-1, legal-action share of predictions
(1.0 by construction under the mask), and honest baselines:
  * legal-uniform: mean over samples of 1/|legal actions|
  * majority-pass: share of labels equal to the pass action
PPO_SEMANTICS: OFF_POLICY_AUXILIARY. The pilot checkpoint NEVER enters the
training funnel or gameplay evaluation (plan consumption gate).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
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

from generals_bot.competition_native_jax.competition_env_jax import (  # noqa: E402
    empty_memory,
    legal_mask_one_jax,
    observe_one_jax,
)
from scripts.data.replay_engine_oracle import state_from_tick  # noqa: E402
from scripts.data.replay_legal_pov import parse_replay  # noqa: E402

ACTION_DIM = 3970
PASS_INDEX = 0


def small_policy(params, spatial, global_vec):
    """Tiny CNN policy over canonical observation tensors -> 3970 logits."""
    x = spatial  # (B, 8, 21, 21)
    for w, b in zip(params["conv_w"], params["conv_b"], strict=True):
        x = jax.nn.relu(
            jax.lax.conv_general_dilated(
                x, w, window_strides=(1, 1), padding="SAME",
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            ) + b[None, :, None, None]
        )
    flat = x.reshape(x.shape[0], -1)
    feat = jnp.concatenate([flat, global_vec], axis=1)
    hidden = jax.nn.relu(feat @ params["fc_w"] + params["fc_b"])
    logits = hidden @ params["out_w"] + params["out_b"]  # (B, 3970)
    return logits


def init_params(key, channels=(32, 32)):
    keys = jax.random.split(key, 6)
    conv_w, conv_b = [], []
    in_c = 8
    for k, out_c in zip(keys[:2], channels, strict=True):
        conv_w.append(jax.random.normal(k, (out_c, in_c, 3, 3)) * 0.1)
        conv_b.append(jnp.zeros(out_c))
        in_c = out_c
    flat_dim = channels[-1] * 21 * 21 + 8
    fc_w = jax.random.normal(keys[2], (flat_dim, 128)) * 0.05
    out_w = jax.random.normal(keys[3], (128, ACTION_DIM)) * 0.01
    return {
        "conv_w": conv_w,
        "conv_b": conv_b,
        "fc_w": fc_w,
        "fc_b": jnp.zeros(128),
        "out_w": out_w,
        "out_b": jnp.zeros(ACTION_DIM),
    }


def build_features(shard_dir: Path, dataset_dir: Path) -> dict[str, dict]:
    """Walk replays sequentially, rebuilding fog memory per seat (legal path)."""
    samples_path = shard_dir / "samples.jsonl"
    by_key: dict[tuple, dict] = {}
    replays_needed: set[str] = set()
    with samples_path.open(encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            by_key[(sample["replay"], sample["tick"], sample["seat"])] = sample
            replays_needed.add(sample["replay"])
    features: dict[str, dict] = {
        "train": defaultdict(list),
        "holdout_player": defaultdict(list),
        "holdout_replay": defaultdict(list),
    }
    raw_dir = dataset_dir / "raw"
    for replay_name in sorted(replays_needed):
        payload = json.loads((raw_dir / replay_name).read_text(encoding="utf-8"))
        replay = parse_replay(payload)
        memories = {seat: empty_memory() for seat in range(2)}
        for t in range(len(replay.ticks)):
            state = state_from_tick(
                replay.ticks[t],
                dims=replay.dims,
                mountains=replay.mountains,
                castles=replay.cities,
                generals=replay.generals,
                time=t,
            )
            for seat in range(2):
                sample = by_key.get((replay_name, t, seat))
                if sample is None:
                    memories[seat] = observe_one_jax(state.engine_state, seat, memories[seat])[2]
                    continue
                spatial, global_vec, memories[seat] = observe_one_jax(
                    state.engine_state, seat, memories[seat]
                )
                mask = legal_mask_one_jax(state.engine_state, seat)
                features[sample["split"]]["spatial"].append(np.asarray(spatial))
                features[sample["split"]]["global"].append(np.asarray(global_vec))
                features[sample["split"]]["mask"].append(np.asarray(mask))
                features[sample["split"]]["label"].append(int(sample["label"]))
    return {
        split: {k: np.stack(v) for k, v in arrays.items()}
        for split, arrays in features.items()
    }


def masked_accuracy(logits, masks, labels) -> tuple[float, float]:
    masked = jnp.where(masks.astype(bool), logits, jnp.finfo(jnp.float32).min)
    preds = jnp.argmax(masked, axis=1)
    acc = float(jnp.mean((preds == labels).astype(jnp.float32)))
    legal_share = float(jnp.mean(jnp.take_along_axis(masks, preds[:, None], axis=1)))
    return acc, legal_share


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("var/marathon_takeover/bc_a_pilot"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    started = time.perf_counter()
    shard_dir = Path(args.shard_dir)
    manifest = json.loads((shard_dir / "shards.json").read_text(encoding="utf-8"))
    print("building features via canonical legal observation path ...")
    feats = build_features(shard_dir, Path(args.dataset_dir))
    train = feats["train"]
    n = train["label"].shape[0]
    print(f"features ready: train={n} holdout_player={feats['holdout_player']['label'].shape[0]} "
          f"holdout_replay={feats['holdout_replay']['label'].shape[0]} "
          f"({time.perf_counter() - started:.1f}s)")

    key = jax.random.PRNGKey(args.seed)
    key, pk = jax.random.split(key)
    params = init_params(pk)
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, spatial, global_vec, masks, labels):
        logits = small_policy(params, spatial, global_vec)
        masked = jnp.where(masks.astype(bool), logits, jnp.finfo(jnp.float32).min)
        logp = jax.nn.log_softmax(masked, axis=1)
        return -jnp.mean(jnp.take_along_axis(logp, labels[:, None], axis=1))

    @jax.jit
    def step(params, opt_state, spatial, global_vec, masks, labels):
        loss, grads = jax.value_and_grad(loss_fn)(params, spatial, global_vec, masks, labels)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state_new, loss

    rng = np.random.default_rng(args.seed)
    history = []
    for _ in range(args.epochs):
        order = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            params, opt_state, loss = step(
                params, opt_state,
                jnp.asarray(train["spatial"][idx]),
                jnp.asarray(train["global"][idx]),
                jnp.asarray(train["mask"][idx]),
                jnp.asarray(train["label"][idx]),
            )
            epoch_loss += float(loss)
        history.append(epoch_loss / max(1, n // args.batch_size))

    results = {"kind": "BC_A_PILOT_RESULT", "seed": args.seed, "epochs": args.epochs,
               "shard_dataset_id": manifest["dataset_id"],
               "samples_sha256": manifest["samples_sha256"],
               "engine_sha": manifest["engine_sha"],
               "wall_s": round(time.perf_counter() - started, 1),
               "loss_first": history[0], "loss_last": history[-1]}
    eval_splits = {
        "train": train,
        "holdout_player": feats["holdout_player"],
        "holdout_replay": feats["holdout_replay"],
    }
    for split_name, split in eval_splits.items():
        logits = small_policy(
            params, jnp.asarray(split["spatial"]), jnp.asarray(split["global"])
        )
        acc, legal_share = masked_accuracy(
            logits, jnp.asarray(split["mask"]), jnp.asarray(split["label"])
        )
        labels = split["label"]
        masks = split["mask"]
        legal_counts = masks.sum(axis=1)
        uniform_baseline = float(np.mean(1.0 / np.maximum(legal_counts, 1)))
        pass_baseline = float(np.mean(labels == PASS_INDEX))
        results[split_name] = {
            "top1_accuracy": round(acc, 6),
            "legal_action_share": round(legal_share, 6),
            "legal_uniform_baseline": round(uniform_baseline, 6),
            "majority_pass_baseline": round(pass_baseline, 6),
            "n": int(labels.shape[0]),
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "bc_a_pilot_result.json"
    out_path.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
