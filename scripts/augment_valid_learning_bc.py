#!/usr/bin/env python3
"""Add one isolated all-four-opponent validation round to a BC corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from collect_valid_learning_bc import (
    OPPONENTS,
    TEACHER_CANONICAL_ID,
    TEACHER_ID,
    collection_source_hashes,
    file_sha256,
    sample_identity,
)
from generals.core import game

from generals_bot.competition_native_jax.action_codec import action_to_index
from generals_bot.competition_native_jax.competition_env_jax import (
    TRUNCATION,
    competition_transition,
    empty_memory,
    legal_mask_one_jax,
    observe_one_jax,
    reset_one_jax,
)
from generals_bot.observation import GameContext
from generals_bot.policies.base import TraceLevel
from generals_bot.selector import create_policy
from generals_bot.training.bridge_benchmark import extract_numpy_boards
from generals_bot.training.collect_bc import _action_to_jax, _observation_from_arrays

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path.home() / "quantsilico-runtime" / "cloud_valid_learning_recovery_v1"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=50_000)
    parser.add_argument("--max-turns", type=int, default=1_200)
    args = parser.parse_args()
    base_manifest = json.loads(
        args.base.with_name("dataset_manifest.json").read_text(encoding="utf-8")
    )
    if file_sha256(args.base) != base_manifest["dataset_sha256"]:
        raise RuntimeError("base BC dataset SHA mismatch")
    provenance = json.loads(
        (ROOT / "experiments/manifests/bc_teacher_fallback_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    raw = np.load(args.base, allow_pickle=False)
    rows = {name: [item for item in raw[name]] for name in raw.files}
    rows["split"] = ["train"] * len(rows["teacher_action"])
    seen = set(str(item) for item in rows["sample_id"])
    game_reports: list[dict] = []
    started = time.perf_counter()

    for game_index, (opponent_name, learner_seat) in enumerate(
        (opponent, seat) for opponent in OPPONENTS for seat in (0, 1)
    ):
        seed = args.seed_base + game_index
        game_id = f"{opponent_name}:seat{learner_seat}:seed{seed}"
        state = reset_one_jax(jax.random.PRNGKey(seed), 21, 21)
        teacher = create_policy(TEACHER_CANONICAL_ID)
        opponent = create_policy(opponent_name, seed=seed + 100_003)
        teacher_state = teacher.initial_state(GameContext(learner_seat, 21, 21))
        opponent_seat = 1 - learner_seat
        opponent_state = opponent.initial_state(GameContext(opponent_seat, 21, 21))
        native_memory = empty_memory()
        before = len(rows["teacher_action"])
        winner = -1
        turns = 0
        for _ in range(min(args.max_turns, TRUNCATION)):
            learner_engine_obs = game.get_observation(state, learner_seat)
            opponent_engine_obs = game.get_observation(state, opponent_seat)
            lt, lo, la, _lg, lm = extract_numpy_boards(learner_engine_obs, 21, 21)
            ot, oo, oa, _og, om = extract_numpy_boards(opponent_engine_obs, 21, 21)
            learner_obs = _observation_from_arrays(lt, lo, la, lm)
            opponent_obs = _observation_from_arrays(ot, oo, oa, om)
            teacher_decision = teacher.act(
                learner_obs,
                teacher_state,
                deterministic=True,
                trace=TraceLevel.NONE,
                deadline=None,
            )
            opponent_decision = opponent.act(
                opponent_obs,
                opponent_state,
                deterministic=True,
                trace=TraceLevel.NONE,
                deadline=None,
            )
            teacher_state = teacher_decision.new_state
            opponent_state = opponent_decision.new_state
            spatial, global_vec, native_memory = observe_one_jax(
                state, learner_seat, native_memory
            )
            legal_mask = legal_mask_one_jax(state, learner_seat)
            teacher_action = action_to_index(teacher_decision.action)
            if not bool(np.asarray(legal_mask)[teacher_action]):
                raise RuntimeError(f"illegal teacher target: {game_id} turn={int(state.time)}")
            spatial_np = np.asarray(spatial, dtype=np.float16)
            global_np = np.asarray(global_vec, dtype=np.float32)
            seen_np = np.asarray(native_memory.seen_own, dtype=np.float16)
            army_np = np.asarray(native_memory.last_army, dtype=np.float16)
            legal_np = np.asarray(legal_mask, dtype=bool)
            identity = sample_identity(
                spatial_np, global_np, seen_np, army_np, legal_np, learner_seat
            )
            if identity not in seen:
                seen.add(identity)
                values = {
                    "spatial": spatial_np,
                    "global_vec": global_np,
                    "memory_seen_own": seen_np,
                    "memory_last_army": army_np,
                    "legal_mask": legal_np,
                    "teacher_action": teacher_action,
                    "opponent": opponent_name,
                    "seat": learner_seat,
                    "seed": seed,
                    "turn": int(state.time),
                    "game_id": game_id,
                    "split": "validation",
                    "sample_id": identity,
                }
                for name, value in values.items():
                    rows[name].append(value)
            learner_action = _action_to_jax(teacher_decision.action)
            foe_action = _action_to_jax(opponent_decision.action)
            actions = (
                jnp.stack([learner_action, foe_action])
                if learner_seat == 0
                else jnp.stack([foe_action, learner_action])
            )
            state, info = competition_transition(state, actions)
            turns += 1
            if bool(info.is_done):
                winner = int(info.winner)
                break
        game_reports.append(
            {
                "game_id": game_id,
                "opponent": opponent_name,
                "learner_seat": learner_seat,
                "seed": seed,
                "split": "validation",
                "turns": turns,
                "winner": winner,
                "unique_samples": len(rows["teacher_action"]) - before,
            }
        )
        print(f"BC_AUGMENT game={game_id} turns={turns} total={len(seen)}", flush=True)

    identity = hashlib.sha256(
        f"{base_manifest['dataset_sha256']}:{args.seed_base}:all_four_both_seats".encode()
    ).hexdigest()
    output = RUNTIME / "bc" / f"dataset_aug_{identity[:16]}"
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / "teacher_states.npz"
    np.savez_compressed(
        dataset,
        spatial=np.stack(rows["spatial"]),
        global_vec=np.stack(rows["global_vec"]),
        memory_seen_own=np.stack(rows["memory_seen_own"]),
        memory_last_army=np.stack(rows["memory_last_army"]),
        legal_mask=np.stack(rows["legal_mask"]),
        teacher_action=np.asarray(rows["teacher_action"], dtype=np.int32),
        opponent=np.asarray(rows["opponent"], dtype="U24"),
        seat=np.asarray(rows["seat"], dtype=np.int8),
        seed=np.asarray(rows["seed"], dtype=np.int32),
        turn=np.asarray(rows["turn"], dtype=np.int32),
        game_id=np.asarray(rows["game_id"], dtype="U64"),
        split=np.asarray(rows["split"], dtype="U10"),
        sample_id=np.asarray(rows["sample_id"], dtype="U32"),
    )
    split = np.asarray(rows["split"])
    report = {
        "schema_version": 1,
        "kind": "VALID_LEARNING_BC_DATASET_AUGMENTED",
        "status": "PASS",
        "dataset_id": f"BC_VALID_LEARNING_{file_sha256(dataset)[:16]}",
        "dataset": str(dataset),
        "dataset_sha256": file_sha256(dataset),
        "unique_recurrent_states": len(seen),
        "base_dataset_sha256": base_manifest["dataset_sha256"],
        "teacher": {
            "BC_TEACHER_ID": TEACHER_ID,
            "BC_TEACHER_SOURCE_HASH": provenance["BC_TEACHER_SOURCE_HASH"],
            "BC_TEACHER_EQUALS_FALLBACK": provenance["BC_TEACHER_EQUALS_FALLBACK"],
        },
        "source_hashes": collection_source_hashes(),
        "coverage": {
            "opponents": sorted(set(str(item) for item in rows["opponent"])),
            "seats": sorted(set(int(item) for item in rows["seat"])),
            "splits": {
                "train": int(np.count_nonzero(split == "train")),
                "validation": int(np.count_nonzero(split == "validation")),
            },
            "train_games": len(set(np.asarray(rows["game_id"])[split == "train"])),
            "validation_games": len(set(np.asarray(rows["game_id"])[split == "validation"])),
        },
        "validation_game_reports": game_reports,
        "split": "16 complete base games train; 8 new complete games validation",
        "elapsed_s": time.perf_counter() - started,
        "written_at": datetime.now(UTC).isoformat(),
    }
    atomic_json(output / "dataset_manifest.json", report)
    atomic_json(ROOT / "experiments/manifests/valid_learning_bc_dataset_augmented.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
