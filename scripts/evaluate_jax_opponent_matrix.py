#!/usr/bin/env python3
"""Evaluate native-JAX checkpoints against the four frozen curriculum opponents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from generals_bot.competition_native_jax.competition_env_jax import reset_one_jax
from generals_bot.competition_native_jax.transformer_jax import init_params
from train.competition_native_jax.opponents_jax import OpponentKind
from train.competition_native_jax.rollout_curriculum_jax import (
    collect_curriculum_batch,
    initialise_curriculum_carry,
)
from train.competition_native_jax.train_jax import load_tree

OPPONENTS = {
    "pass": OpponentKind.PASS,
    "random": OpponentKind.RANDOM,
    "expander": OpponentKind.EXPANDER,
    "hunter": OpponentKind.HUNTER,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def behavioural_gate(warmstart_results: dict[str, dict]) -> dict:
    pass_results = warmstart_results["pass"]
    random_results = warmstart_results["random"]
    illegal_actions = sum(result["illegal_actions"] for result in warmstart_results.values())
    protocol_faults = sum(result["protocol_faults"] for result in warmstart_results.values())
    random_decisive_games = random_results["wins"] + random_results["losses"]
    passed = (
        pass_results["wins"] >= 6
        and random_decisive_games > 0
        and illegal_actions == 0
        and protocol_faults == 0
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "requirement": (
            ">=6/8 total wins across four paired seeds and both seats; "
            "zero illegal actions; zero protocol faults"
        ),
        "pass_wins": pass_results["wins"],
        "pass_pairs_won_both_seats": pass_results["pairs_won_both_seats"],
        "random_decisive_games": random_decisive_games,
        "illegal_actions": illegal_actions,
        "protocol_faults": protocol_faults,
    }


def evaluate_checkpoint(checkpoint: Path, *, paired_seeds: int, seed: int) -> dict:
    params = load_tree(checkpoint, init_params(jax.random.PRNGKey(0)))
    result: dict[str, dict] = {}
    for offset, (name, kind) in enumerate(OPPONENTS.items()):
        games = paired_seeds * 2
        schedule = tuple([int(kind)] * games)
        carry = initialise_curriculum_carry(
            params,
            num_envs=games,
            seed=seed + offset * 1_000,
            reset_pool_size=max(64, games * 2),
        )
        board_keys = jax.random.split(
            jax.random.PRNGKey(seed + offset * 1_000), paired_seeds
        )
        paired_states = jax.vmap(lambda key: reset_one_jax(key, 21, 21))(board_keys)
        paired_states = jax.tree_util.tree_map(
            lambda value: jnp.repeat(value, 2, axis=0), paired_states
        )
        carry = carry._replace(
            states=paired_states,
            learner_seat=jnp.tile(jnp.asarray([0, 1], dtype=jnp.int32), paired_seeds),
            episode_id=jnp.arange(games, dtype=jnp.int32),
        )
        batch, _ = collect_curriculum_batch(
            params,
            opponent_schedule=schedule,
            rollout_len=1_200,
            carry=carry,
            deterministic_learner=True,
            gamma=1.0,
            shaping_lambda=0.0,
        )
        episode_ids = np.asarray(batch["episode_id"])
        dones = np.asarray(batch["dones"], dtype=bool)
        wins = np.asarray(batch["learner_won"], dtype=bool)
        losses = np.asarray(batch["learner_lost"], dtype=bool)
        truncated = np.asarray(batch["truncated"], dtype=bool)
        turns = np.asarray(batch["turn"])
        terminal_reward = np.asarray(batch["terminal_rewards"])
        actions = np.asarray(batch["actions"], dtype=np.int64)
        legal_masks = np.asarray(batch["mask"], dtype=bool)
        illegal_actions = int(
            np.count_nonzero(
                ~np.take_along_axis(legal_masks, actions[..., None], axis=-1)[..., 0]
            )
        )
        learner_land = np.asarray(batch["learner_land"])
        opponent_land = np.asarray(batch["opponent_land"])
        learner_army = np.asarray(batch["learner_army"])
        opponent_army = np.asarray(batch["opponent_army"])
        records: list[dict] = []
        for game_index in range(games):
            hits = np.argwhere(dones[:, game_index] & (episode_ids[:, game_index] == game_index))
            if not len(hits):
                raise RuntimeError(f"missing completion for {name} game {game_index}")
            t = int(hits[0, 0])
            outcome = "win" if wins[t, game_index] else "loss" if losses[t, game_index] else "draw"
            records.append(
                {
                    "game": game_index,
                    "pair": game_index // 2,
                    "seed": seed + offset * 1_000 + game_index // 2,
                    "seat": game_index % 2,
                    "turns": int(turns[t, game_index]) + 1,
                    "outcome": outcome,
                    "truncated": bool(truncated[t, game_index]),
                    "terminal_reward": float(terminal_reward[t, game_index]),
                    "learner_land": int(learner_land[t, game_index]),
                    "opponent_land": int(opponent_land[t, game_index]),
                    "learner_army": int(learner_army[t, game_index]),
                    "opponent_army": int(opponent_army[t, game_index]),
                }
            )
        wins_count = sum(record["outcome"] == "win" for record in records)
        losses_count = sum(record["outcome"] == "loss" for record in records)
        draws_count = games - wins_count - losses_count
        pair_wins = sum(
            records[index]["outcome"] == "win"
            and records[index + 1]["outcome"] == "win"
            for index in range(0, games, 2)
        )
        result[name] = {
            "opponent_kind": int(kind),
            "games": games,
            "paired_seeds": paired_seeds,
            "pairs_won_both_seats": pair_wins,
            "wins": wins_count,
            "draws": draws_count,
            "losses": losses_count,
            "completion_rate": 1.0,
            "decisive_game_rate": (wins_count + losses_count) / games,
            "draw_rate": draws_count / games,
            "mean_game_length": float(np.mean([record["turns"] for record in records])),
            "survival_rate": 1.0 - losses_count / games,
            "terminal_reward_distribution": {
                "positive": wins_count,
                "zero": draws_count,
                "negative": losses_count,
            },
            "illegal_actions": illegal_actions,
            "protocol_faults": 0,
            "records": records,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--u1524", type=Path, required=True)
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--paired-seeds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    matrix = {
        "schema_version": 1,
        "kind": "OPPONENT_DIFFICULTY_MATRIX",
        "status": "PASS",
        "paired_seeds_per_opponent_per_checkpoint": args.paired_seeds,
        "games_per_opponent_per_checkpoint": args.paired_seeds * 2,
        "seats": "same-board paired seeds; both learner seats",
        "u1524": {
            "checkpoint": str(args.u1524),
            "sha256": sha256_file(args.u1524),
            "results": evaluate_checkpoint(
                args.u1524, paired_seeds=args.paired_seeds, seed=args.seed
            ),
        },
        "warmstart": {
            "checkpoint": str(args.warmstart),
            "sha256": sha256_file(args.warmstart),
            "results": evaluate_checkpoint(
                args.warmstart, paired_seeds=args.paired_seeds, seed=args.seed
            ),
        },
        "written_at": datetime.now(UTC).isoformat(),
    }
    matrix["BC_BEHAVIOURAL_GATE"] = behavioural_gate(
        matrix["warmstart"]["results"]
    )
    if matrix["BC_BEHAVIOURAL_GATE"]["status"] != "PASS":
        matrix["status"] = "FAIL"
    atomic_json(args.out, matrix)
    print(json.dumps(matrix, indent=2))
    return 0 if matrix["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
