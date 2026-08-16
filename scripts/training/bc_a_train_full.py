"""BC-A-FULL-R1 trainer (predeclared: bc_a_full_round_1_plan.yaml).

Same pipeline as the pilot (bc_a_train_pilot.py, EV-0050) adapted to the
union corpus: samples carry dataset/player/phase; features reconstructed ONLY
through the canonical legal observation path; metrics reported per holdout
player SEPARATELY and phase-stratified. Seed 20260831 per predeclaration.
PPO_SEMANTICS: OFF_POLICY_AUXILIARY. Consumption gate: a passing checkpoint
is eligible only as warm-start input for a separately predeclared
PPO-continuation sub-experiment; BC accuracy never promotes.
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
from scripts.training.bc_a_train_pilot import (  # noqa: E402
    PASS_INDEX,
    init_params,
    masked_accuracy,
    small_policy,
)

ELITE_ROOT = REPO / "experiments/datasets/elite_replays"
PHASES = ("0-199", "200-399", "400-799", "800+")


def build_features(shard_dir: Path) -> dict[str, dict]:
    """Walk replays sequentially, rebuilding fog memory per seat (legal path)."""
    samples_path = shard_dir / "samples.jsonl"
    by_key: dict[tuple, dict] = {}
    replays_needed: dict[str, set[str]] = defaultdict(set)
    with samples_path.open(encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            by_key[(sample["dataset"], sample["replay"], sample["tick"], sample["seat"])] = sample
            replays_needed[sample["dataset"]].add(sample["replay"])
    features: dict[str, dict] = {
        split: defaultdict(list) for split in ("train", "holdout_player", "holdout_replay")
    }
    done = 0
    total = sum(len(v) for v in replays_needed.values())
    for dataset, names in sorted(replays_needed.items()):
        raw_dir = ELITE_ROOT / dataset / "raw"
        for replay_name in sorted(names):
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
                    sample = by_key.get((dataset, replay_name, t, seat))
                    if sample is None:
                        memories[seat] = observe_one_jax(
                            state.engine_state, seat, memories[seat]
                        )[2]
                        continue
                    spatial, global_vec, memories[seat] = observe_one_jax(
                        state.engine_state, seat, memories[seat]
                    )
                    mask = legal_mask_one_jax(state.engine_state, seat)
                    bucket = features[sample["split"]]
                    bucket["spatial"].append(np.asarray(spatial, dtype=np.float16))
                    bucket["global"].append(np.asarray(global_vec, dtype=np.float16))
                    bucket["mask"].append(np.asarray(mask, dtype=bool))
                    bucket["label"].append(int(sample["label"]))
                    bucket["player"].append(sample["player"])
                    bucket["phase"].append(sample["phase"])
            done += 1
            if done % 10 == 0:
                print(f"features: {done}/{total} replays", flush=True)
    return {
        split: {
            k: (np.stack(v) if k not in ("player", "phase") else list(v))
            for k, v in arrays.items()
        }
        for split, arrays in features.items()
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


def evaluate_subset(params, split: dict, keep: np.ndarray, name: str) -> dict:
    sub = {
        "spatial": split["spatial"][keep],
        "global": split["global"][keep],
        "mask": split["mask"][keep],
        "label": split["label"][keep],
    }
    if int(keep.sum()) == 0:
        return {"n": 0, "split": name}
    return evaluate(params, sub, name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("var/marathon_takeover/bc_a_full"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    started = time.perf_counter()
    shard_dir = Path(args.shard_dir)
    manifest = json.loads((shard_dir / "shards.json").read_text(encoding="utf-8"))
    print("building features via canonical legal observation path ...")
    feats = build_features(shard_dir)
    train = feats["train"]
    n = train["label"].shape[0]
    print(
        f"features ready ({time.perf_counter() - started:.0f}s): train={n} "
        f"holdout_player={feats['holdout_player']['label'].shape[0]} "
        f"holdout_replay={feats['holdout_replay']['label'].shape[0]}"
    )

    key = jax.random.PRNGKey(args.seed)
    key, pk = jax.random.split(key)
    params = init_params(pk)
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, spatial, global_vec, masks, labels):
        logits = small_policy(params, spatial, global_vec)
        masked = jnp.where(masks, logits, jnp.finfo(jnp.float32).min)
        logp = jax.nn.log_softmax(masked, axis=1)
        return -jnp.mean(jnp.take_along_axis(logp, labels[:, None], axis=1))

    @jax.jit
    def step(params, opt_state, spatial, global_vec, masks, labels):
        loss, grads = jax.value_and_grad(loss_fn)(params, spatial, global_vec, masks, labels)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state_new, loss

    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(args.epochs):
        order = rng.permutation(n)
        epoch_loss, batches = 0.0, 0
        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            params, opt_state, loss = step(
                params,
                opt_state,
                jnp.asarray(train["spatial"][idx], dtype=jnp.float32),
                jnp.asarray(train["global"][idx], dtype=jnp.float32),
                jnp.asarray(train["mask"][idx]),
                jnp.asarray(train["label"][idx]),
            )
            epoch_loss += float(loss)
            batches += 1
        history.append(epoch_loss / max(1, batches))
        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch + 1}/{args.epochs} loss={history[-1]:.4f}", flush=True)

    results = {
        "kind": "BC_A_FULL_RESULT",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "shard_dataset_id": manifest["dataset_id"],
        "samples_sha256": manifest["samples_sha256"],
        "engine_sha": manifest["engine_sha"],
        "holdout_players": manifest["holdout_players"],
        "wall_s": round(time.perf_counter() - started, 1),
        "loss_first": history[0],
        "loss_last": history[-1],
    }
    results["train"] = evaluate(params, train, "train")
    hp = feats["holdout_player"]
    results["holdout_player_all"] = evaluate(params, hp, "holdout_player_all")
    players = np.asarray(hp["player"])
    for holdout in manifest["holdout_players"]:
        results[f"holdout_player_{holdout}"] = evaluate_subset(
            params, hp, players == holdout, f"holdout_player_{holdout}"
        )
    results["holdout_replay"] = evaluate(params, feats["holdout_replay"], "holdout_replay")
    gen_splits = {
        "holdout_player": hp,
        "holdout_replay": feats["holdout_replay"],
    }
    for split_name, split in gen_splits.items():
        phases = np.asarray(split["phase"])
        for phase in PHASES:
            results[f"phase_{split_name}_{phase}"] = evaluate_subset(
                params, split, phases == phase, f"phase_{split_name}_{phase}"
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "bc_a_full_result.json"
    out_path.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
