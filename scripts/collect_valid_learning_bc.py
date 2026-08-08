#!/usr/bin/env python3
"""Collect unique native-JAX BC states from the frozen heuristic teacher."""

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
TEACHER_ID = "heuristic_v2f_plus_planner_terminal_form"
TEACHER_CANONICAL_ID = "heuristic_v2f_plus_planner_terminal_fix"
OPPONENTS = ("pass", "legal_random", "official_expander", "official_hunter")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_bundle_sha256(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        relative = path.relative_to(ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def collection_source_hashes() -> dict[str, str]:
    official = ROOT / "third_party" / "generals-bots" / "generals" / "agents"
    return {
        "BC_RUNTIME_SOURCE_HASH": source_bundle_sha256(
            (
                ROOT / "src/generals_bot/competition_native_jax/action_codec.py",
                ROOT / "src/generals_bot/competition_native_jax/competition_env_jax.py",
                ROOT / "src/generals_bot/training/bridge_benchmark.py",
                ROOT / "src/generals_bot/training/collect_bc.py",
            )
        ),
        "BC_OPPONENT_SOURCE_HASH": source_bundle_sha256(
            (
                official / "random_agent.py",
                official / "expander_agent.py",
                official / "hunter_agent.py",
            )
        ),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sample_identity(
    spatial: np.ndarray,
    global_vec: np.ndarray,
    seen_own: np.ndarray,
    last_army: np.ndarray,
    legal_mask: np.ndarray,
    seat: int,
) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for value in (spatial, global_vec, seen_own, last_army, legal_mask):
        digest.update(np.ascontiguousarray(value).view(np.uint8))
    digest.update(bytes([seat]))
    return digest.hexdigest()


def collect(args: argparse.Namespace) -> dict:
    provenance = json.loads(Path(args.provenance).read_text(encoding="utf-8"))
    if provenance.get("status") != "PASS":
        raise RuntimeError("HEURISTIC_BASELINE_PROVENANCE must pass before collection")

    source_hashes = collection_source_hashes()
    config = {
        "teacher": TEACHER_ID,
        "teacher_canonical": TEACHER_CANONICAL_ID,
        "opponents": OPPONENTS,
        "rounds": args.rounds,
        "seed_base": args.seed_base,
        "target_states": args.target_states,
        "minimum_states": args.minimum_states,
        "max_turns": args.max_turns,
        "board": [21, 21],
        "split": "complete_game_and_seed_round0_train_round1_validation",
        "teacher_source_hash": provenance["BC_TEACHER_SOURCE_HASH"],
        **source_hashes,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output = RUNTIME / "bc" / f"dataset_{config_hash[:16]}"
    if output.exists():
        raise FileExistsError(f"refusing to mix/overwrite dataset directory: {output}")
    output.mkdir(parents=True)

    rows: dict[str, list] = {
        "spatial": [],
        "global_vec": [],
        "memory_seen_own": [],
        "memory_last_army": [],
        "legal_mask": [],
        "teacher_action": [],
        "opponent": [],
        "seat": [],
        "seed": [],
        "turn": [],
        "game_id": [],
        "split": [],
        "sample_id": [],
    }
    seen: set[str] = set()
    game_reports: list[dict] = []
    started = time.perf_counter()

    jobs = [
        (round_index, opponent, seat)
        for round_index in range(args.rounds)
        for opponent in OPPONENTS
        for seat in (0, 1)
    ]
    for game_index, (round_index, opponent_name, learner_seat) in enumerate(jobs):
        if time.perf_counter() - started > args.wall_minutes * 60:
            break
        seed = args.seed_base + game_index
        split = "validation" if round_index == args.rounds - 1 else "train"
        game_id = f"{opponent_name}:seat{learner_seat}:seed{seed}"
        state = reset_one_jax(jax.random.PRNGKey(seed), 21, 21)
        teacher = create_policy(TEACHER_CANONICAL_ID)
        opponent = create_policy(opponent_name, seed=seed + 100_003)
        teacher_state = teacher.initial_state(GameContext(learner_seat, 21, 21))
        opponent_seat = 1 - learner_seat
        opponent_state = opponent.initial_state(GameContext(opponent_seat, 21, 21))
        native_memory = empty_memory()
        before_count = len(rows["teacher_action"])
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
                raise RuntimeError(
                    f"illegal teacher action game={game_id} turn={int(state.time)} "
                    f"action={teacher_decision.action} index={teacher_action}"
                )
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
                rows["spatial"].append(spatial_np)
                rows["global_vec"].append(global_np)
                rows["memory_seen_own"].append(seen_np)
                rows["memory_last_army"].append(army_np)
                rows["legal_mask"].append(legal_np)
                rows["teacher_action"].append(teacher_action)
                rows["opponent"].append(opponent_name)
                rows["seat"].append(learner_seat)
                rows["seed"].append(seed)
                rows["turn"].append(int(state.time))
                rows["game_id"].append(game_id)
                rows["split"].append(split)
                rows["sample_id"].append(identity)

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
                "split": split,
                "turns": turns,
                "winner": winner,
                "unique_samples": len(rows["teacher_action"]) - before_count,
            }
        )
        print(
            f"BC_COLLECT game={game_id} turns={turns} "
            f"unique={len(rows['teacher_action'])}",
            flush=True,
        )

    unique_count = len(rows["teacher_action"])
    if unique_count < args.minimum_states:
        raise RuntimeError(
            f"unique recurrent state gate failed: {unique_count} < {args.minimum_states}"
        )
    if set(rows["opponent"]) != set(OPPONENTS) or set(rows["seat"]) != {0, 1}:
        raise RuntimeError("all-four-opponent/both-seat dataset coverage gate failed")
    if set(rows["split"]) != {"train", "validation"}:
        raise RuntimeError("complete-game train/validation split coverage failed")

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
    dataset_sha = file_sha256(dataset)
    report = {
        "schema_version": 1,
        "kind": "VALID_LEARNING_BC_DATASET",
        "status": "PASS",
        "dataset_id": f"BC_VALID_LEARNING_{dataset_sha[:16]}",
        "dataset": str(dataset),
        "dataset_sha256": dataset_sha,
        "unique_recurrent_states": unique_count,
        "target_states": args.target_states,
        "minimum_states": args.minimum_states,
        "teacher": {
            "BC_TEACHER_ID": provenance["BC_TEACHER_ID"],
            "BC_TEACHER_SOURCE_HASH": provenance["BC_TEACHER_SOURCE_HASH"],
            "BC_TEACHER_EQUALS_FALLBACK": provenance["BC_TEACHER_EQUALS_FALLBACK"],
        },
        "source_hashes": source_hashes,
        "coverage": {
            "opponents": sorted(set(rows["opponent"])),
            "seats": sorted(set(rows["seat"])),
            "splits": {
                name: int(sum(item == name for item in rows["split"]))
                for name in ("train", "validation")
            },
        },
        "game_reports": game_reports,
        "config": config,
        "config_hash": config_hash,
        "elapsed_s": time.perf_counter() - started,
        "written_at": datetime.now(UTC).isoformat(),
    }
    atomic_json(output / "dataset_manifest.json", report)
    atomic_json(ROOT / "experiments" / "manifests" / "valid_learning_bc_dataset.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provenance",
        type=Path,
        default=ROOT / "experiments" / "manifests" / "bc_teacher_fallback_provenance.json",
    )
    parser.add_argument("--target-states", type=int, default=20_000)
    parser.add_argument("--minimum-states", type=int, default=12_000)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=30_000)
    parser.add_argument("--max-turns", type=int, default=1_200)
    parser.add_argument("--wall-minutes", type=float, default=50.0)
    return parser.parse_args()


def main() -> int:
    report = collect(parse_args())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
