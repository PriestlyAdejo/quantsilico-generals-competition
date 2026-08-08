#!/usr/bin/env python3
"""Bounded closed-loop diagnosis and targeted DAgger collection for the noon rescue."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from generals.core import game

from generals_bot.competition_native_jax.action_codec import action_to_index
from generals_bot.competition_native_jax.competition_env_jax import (
    TRUNCATION,
    ObsMemoryJax,
    competition_transition,
    empty_memory,
    index_to_engine_action,
    legal_mask_one_jax,
    observe_one_jax,
    reset_one_jax,
    step_batch_jax,
)
from generals_bot.competition_native_jax.inference_jax import masked_log_softmax
from generals_bot.competition_native_jax.transformer_jax import (
    forward_batch,
    forward_jit,
    init_params,
)
from generals_bot.observation import GameContext
from generals_bot.policies.base import PolicyState, TraceLevel
from generals_bot.selector import create_policy
from generals_bot.training.bridge_benchmark import extract_numpy_boards
from generals_bot.training.collect_bc import _action_to_jax, _observation_from_arrays
from train.competition_native_jax.opponents_jax import (
    OpponentKind,
    batched_opponent_actions,
)
from train.competition_native_jax.rollout_curriculum_jax import (
    CurriculumCarry,
    initialise_curriculum_carry,
    rollout_step,
)
from train.competition_native_jax.train_jax import load_tree

ROOT = Path(__file__).resolve().parents[1]
TEACHER_ID = "heuristic_v2f_plus_planner_terminal_fix"
CHECKPOINT_TURNS = (64, 128, 256, 512, 800)
OPPONENTS = ("pass", "legal_random")


@dataclass
class Snapshot:
    turn: int
    state: Any
    memory: ObsMemoryJax
    teacher: Any
    teacher_state: PolicyState
    opponent_kind: int
    rollout_key: Any
    opponent_env_index: int
    opponent_num_envs: int
    learner_seat: int
    game_id: str


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


def clone_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: jnp.array(value), tree)


def tree_equal(left: Any, right: Any) -> bool:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    return len(left_leaves) == len(right_leaves) and all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def engine_observation(state: Any, player: int):
    raw = game.get_observation(state, player)
    h, w = state.armies.shape
    type_grid, owner, armies, _globals, meta = extract_numpy_boards(raw, h, w)
    return _observation_from_arrays(type_grid, owner, armies, meta), meta


def policy_action(policy: Any, observation: Any, state: PolicyState):
    decision = policy.act(
        observation,
        state,
        deterministic=True,
        trace=TraceLevel.NONE,
        deadline=None,
    )
    return decision.action, decision.new_state


def student_inference(params: dict, state: Any, seat: int, memory: ObsMemoryJax):
    spatial, global_vec, memory = observe_one_jax(state, seat, memory)
    mask = legal_mask_one_jax(state, seat)
    output = forward_jit(params, spatial, global_vec)
    logp = masked_log_softmax(output["flat_logits"], mask)
    action = jnp.argmax(jnp.where(mask, logp, jnp.finfo(jnp.float32).min))
    action, logp = jax.device_get((action, logp))
    return int(action), np.asarray(logp), spatial, global_vec, mask, memory


def action_kind(index: int) -> str:
    if index == 0:
        return "pass"
    return "build" if (index - 1) % 9 == 8 else "move"


def phase_for(turn: int) -> str:
    if turn < 128:
        return "opening"
    if turn < 512:
        return "midgame"
    return "conversion"


def sample_identity(
    spatial: np.ndarray,
    global_vec: np.ndarray,
    memory: ObsMemoryJax,
    legal_mask: np.ndarray,
    seat: int,
) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for value in (
        spatial,
        global_vec,
        np.asarray(memory.seen_own, dtype=np.float16),
        np.asarray(memory.last_army, dtype=np.float16),
        legal_mask,
    ):
        digest.update(np.ascontiguousarray(value).view(np.uint8))
    digest.update(bytes([seat]))
    return digest.hexdigest()


def transition_for_actions(state: Any, learner_seat: int, learner: Any, foe: Any):
    actions = (
        jnp.stack([learner, foe])
        if learner_seat == 0
        else jnp.stack([foe, learner])
    )
    return competition_transition(state, actions)


def scripted_opponent_action(
    state: Any,
    learner_seat: int,
    opponent_kind: int,
    key: Any,
) -> Any:
    batched_state = jax.tree_util.tree_map(lambda value: value[None, ...], state)
    action = batched_opponent_actions(
        batched_state,
        jnp.asarray([1 - learner_seat], dtype=jnp.int32),
        jnp.asarray([key]),
        (opponent_kind,),
    )
    return action[0]


def opponent_action_from_rollout_key(
    state: Any,
    learner_seat: int,
    opponent_kind: int,
    rollout_key: Any,
    *,
    environment_index: int,
    num_environments: int,
) -> tuple[Any, Any]:
    """Reproduce rollout_step's exact opponent key and return its next carry key."""
    next_key, _learner_key, opponent_key, _next_seat_key = jax.random.split(
        rollout_key, 4
    )
    environment_key = jax.random.split(opponent_key, num_environments)[
        environment_index
    ]
    action = scripted_opponent_action(
        state,
        learner_seat,
        opponent_kind,
        environment_key,
    )
    return action, next_key


@partial(
    jax.jit,
    static_argnames=("opponent_schedule", "rollout_len"),
)
def capture_student_rollout_scan(
    carry: CurriculumCarry,
    *,
    opponent_schedule: tuple[int, ...],
    rollout_len: int,
):
    def step(inner: CurriculumCarry, _):
        captured = (
            inner.states,
            inner.mem0,
            inner.mem1,
            inner.learner_seat,
            inner.episode_id,
            inner.key,
        )
        next_carry, trajectory = rollout_step(
            inner,
            None,
            opponent_schedule=opponent_schedule,
            gamma=1.0,
            shaping_lambda=0.0,
            deterministic_learner=True,
        )
        return next_carry, (trajectory, captured)

    return jax.lax.scan(step, carry, xs=None, length=rollout_len)


def capture_student_rollouts(
    params: dict,
    *,
    paired_seeds: int,
    seed_base: int,
) -> tuple[dict[str, Any], tuple[Any, ...], list[dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    states: list[Any] = []
    schedule: list[int] = []
    seats: list[int] = []
    for opponent_index, opponent_name in enumerate(OPPONENTS):
        kind = int(OpponentKind.PASS if opponent_name == "pass" else OpponentKind.RANDOM)
        for pair in range(paired_seeds):
            seed = seed_base + opponent_index * 1_000 + pair
            board = reset_one_jax(jax.random.PRNGKey(seed), 21, 21)
            for seat in (0, 1):
                jobs.append(
                    {
                        "opponent": opponent_name,
                        "opponent_kind": kind,
                        "seed": seed,
                        "seat": seat,
                        "game_id": f"student:{opponent_name}:seat{seat}:seed{seed}",
                    }
                )
                states.append(board)
                schedule.append(kind)
                seats.append(seat)
    num_envs = len(jobs)
    stacked_states = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *states)
    repeats = max(2, (2 * num_envs + num_envs - 1) // num_envs)
    pool = jax.tree_util.tree_map(
        lambda value: jnp.concatenate([value] * repeats, axis=0),
        stacked_states,
    )
    carry = initialise_curriculum_carry(
        params,
        num_envs=num_envs,
        seed=seed_base + 91_002,
        pool=pool,
    )
    carry = carry._replace(
        states=stacked_states,
        learner_seat=jnp.asarray(seats, dtype=jnp.int32),
        episode_id=jnp.arange(num_envs, dtype=jnp.int32),
    )
    _final, (batch, captured) = capture_student_rollout_scan(
        carry,
        opponent_schedule=tuple(schedule),
        rollout_len=TRUNCATION,
    )
    return jax.device_get(batch), jax.device_get(captured), jobs


def append_row(
    rows: dict[str, list[Any]],
    *,
    spatial: Any,
    global_vec: Any,
    memory: ObsMemoryJax,
    legal_mask: Any,
    teacher_index: int,
    student_index: int,
    teacher_rank: int,
    opponent: str,
    seat: int,
    seed: int,
    turn: int,
    game_id: str,
    source: str,
    weight: float,
    divergence_turn: int,
    takeover_outcome: str,
) -> None:
    spatial_np = np.asarray(spatial, dtype=np.float16)
    global_np = np.asarray(global_vec, dtype=np.float32)
    legal_np = np.asarray(legal_mask, dtype=bool)
    identity = sample_identity(spatial_np, global_np, memory, legal_np, seat)
    if identity in rows["_seen"]:
        return
    rows["_seen"].add(identity)
    holdout = int(hashlib.sha256(game_id.encode()).hexdigest(), 16) % 5 == 0
    split = "dagger_holdout" if holdout else "train"
    rows["spatial"].append(spatial_np)
    rows["global_vec"].append(global_np)
    rows["memory_seen_own"].append(np.asarray(memory.seen_own, dtype=np.float16))
    rows["memory_last_army"].append(np.asarray(memory.last_army, dtype=np.float16))
    rows["legal_mask"].append(legal_np)
    rows["teacher_action"].append(teacher_index)
    rows["student_action"].append(student_index)
    rows["teacher_rank"].append(teacher_rank)
    rows["opponent"].append(opponent)
    rows["seat"].append(seat)
    rows["seed"].append(seed)
    rows["turn"].append(turn)
    rows["game_id"].append(game_id)
    rows["split"].append(split)
    rows["sample_id"].append(identity)
    rows["sample_source"].append(source)
    rows["sample_weight"].append(min(float(weight), 3.0))
    rows["phase"].append(phase_for(turn))
    rows["divergence_turn"].append(divergence_turn)
    rows["takeover_outcome"].append(takeover_outcome)


def make_rows() -> dict[str, list[Any]]:
    keys = (
        "spatial",
        "global_vec",
        "memory_seen_own",
        "memory_last_army",
        "legal_mask",
        "teacher_action",
        "student_action",
        "teacher_rank",
        "opponent",
        "seat",
        "seed",
        "turn",
        "game_id",
        "split",
        "sample_id",
        "sample_source",
        "sample_weight",
        "phase",
        "divergence_turn",
        "takeover_outcome",
    )
    return {**{key: [] for key in keys}, "_seen": set()}


def clone_gate(snapshot: Snapshot) -> dict[str, Any]:
    observation, _ = engine_observation(snapshot.state, snapshot.learner_seat)
    teacher_a = copy.deepcopy(snapshot.teacher)
    teacher_b = copy.deepcopy(snapshot.teacher)
    state_a = copy.deepcopy(snapshot.teacher_state)
    state_b = copy.deepcopy(snapshot.teacher_state)
    action_a, _ = policy_action(teacher_a, observation, state_a)
    action_b, _ = policy_action(teacher_b, observation, state_b)
    opponent_a, next_key_a = opponent_action_from_rollout_key(
        snapshot.state,
        snapshot.learner_seat,
        snapshot.opponent_kind,
        snapshot.rollout_key,
        environment_index=snapshot.opponent_env_index,
        num_environments=snapshot.opponent_num_envs,
    )
    opponent_b, next_key_b = opponent_action_from_rollout_key(
        clone_tree(snapshot.state),
        snapshot.learner_seat,
        snapshot.opponent_kind,
        jnp.array(snapshot.rollout_key),
        environment_index=snapshot.opponent_env_index,
        num_environments=snapshot.opponent_num_envs,
    )
    next_a, info_a = transition_for_actions(
        clone_tree(snapshot.state),
        snapshot.learner_seat,
        _action_to_jax(action_a),
        opponent_a,
    )
    next_b, info_b = transition_for_actions(
        clone_tree(snapshot.state),
        snapshot.learner_seat,
        _action_to_jax(action_b),
        opponent_b,
    )
    passed = (
        action_a == action_b
        and np.array_equal(np.asarray(opponent_a), np.asarray(opponent_b))
        and np.array_equal(np.asarray(next_key_a), np.asarray(next_key_b))
        and tree_equal(next_a, next_b)
        and tree_equal(info_a, info_b)
    )
    if not passed:
        raise RuntimeError("TEACHER_SHADOW_CLONE_GATE failed")
    return {
        "status": "TEACHER_SHADOW_CLONE_GATE_PASS",
        "game_id": snapshot.game_id,
        "turn": snapshot.turn,
        "teacher_action": action_to_index(action_a),
        "next_transition_equal": True,
    }


def run_teacher_takeover(
    snapshot: Snapshot,
    params: dict,
    rows: dict[str, list[Any]],
    *,
    seed: int,
    opponent_name: str,
) -> dict[str, Any]:
    state = clone_tree(snapshot.state)
    memory = clone_tree(snapshot.memory)
    teacher = copy.deepcopy(snapshot.teacher)
    teacher_state = copy.deepcopy(snapshot.teacher_state)
    rollout_key = jnp.array(snapshot.rollout_key)
    suffix: list[dict[str, Any]] = []
    winner = -1
    while int(state.time) < TRUNCATION:
        turn = int(state.time)
        learner_obs, _ = engine_observation(state, snapshot.learner_seat)
        teacher_action, teacher_state = policy_action(teacher, learner_obs, teacher_state)
        foe_action, rollout_key = opponent_action_from_rollout_key(
            state,
            snapshot.learner_seat,
            snapshot.opponent_kind,
            rollout_key,
            environment_index=snapshot.opponent_env_index,
            num_environments=snapshot.opponent_num_envs,
        )
        student_index, logp, spatial, global_vec, legal, memory = student_inference(
            params, state, snapshot.learner_seat, memory
        )
        teacher_index = action_to_index(teacher_action)
        legal_np = np.asarray(legal, dtype=bool)
        rank = 1 + int(np.count_nonzero(legal_np & (logp > logp[teacher_index])))
        if (turn - snapshot.turn) % 10 == 0:
            suffix.append(
                {
                    "spatial": spatial,
                    "global_vec": global_vec,
                    "memory": memory,
                    "legal": legal,
                    "teacher": teacher_index,
                    "student": student_index,
                    "rank": rank,
                    "turn": turn,
                }
            )
        state, info = transition_for_actions(
            state,
            snapshot.learner_seat,
            _action_to_jax(teacher_action),
            foe_action,
        )
        if bool(info.is_done):
            winner = int(info.winner)
            break
    outcome = "win" if winner == snapshot.learner_seat else "loss" if winner >= 0 else "draw"
    if outcome == "win":
        for item in suffix[:128]:
            append_row(
                rows,
                spatial=item["spatial"],
                global_vec=item["global_vec"],
                memory=item["memory"],
                legal_mask=item["legal"],
                teacher_index=item["teacher"],
                student_index=item["student"],
                teacher_rank=item["rank"],
                opponent=opponent_name,
                seat=snapshot.learner_seat,
                seed=seed,
                turn=item["turn"],
                game_id=f"takeover:{snapshot.game_id}:t{snapshot.turn}",
                source="takeover_recovery",
                weight=3.0,
                divergence_turn=snapshot.turn,
                takeover_outcome=outcome,
            )
    return {
        "game_id": snapshot.game_id,
        "takeover_turn": snapshot.turn,
        "outcome": outcome,
        "terminal_turn": int(state.time),
        "turns_after_takeover": int(state.time) - snapshot.turn,
        "retained_suffix_samples": min(len(suffix), 128) if outcome == "win" else 0,
    }


def run_student_takeover(snapshot: Snapshot, params: dict) -> dict[str, Any]:
    """Continue one teacher position with the student and the exact opponent RNG."""
    state = clone_tree(snapshot.state)
    memory = clone_tree(snapshot.memory)
    rollout_key = jnp.array(snapshot.rollout_key)
    winner = -1
    while int(state.time) < TRUNCATION:
        student_index, _logp, _spatial, _global, _legal, memory = student_inference(
            params,
            state,
            snapshot.learner_seat,
            memory,
        )
        foe_action, rollout_key = opponent_action_from_rollout_key(
            state,
            snapshot.learner_seat,
            snapshot.opponent_kind,
            rollout_key,
            environment_index=snapshot.opponent_env_index,
            num_environments=snapshot.opponent_num_envs,
        )
        state, info = transition_for_actions(
            state,
            snapshot.learner_seat,
            index_to_engine_action(jnp.int32(student_index)),
            foe_action,
        )
        if bool(info.is_done):
            winner = int(info.winner)
            break
    outcome = "win" if winner == snapshot.learner_seat else "loss" if winner >= 0 else "draw"
    return {
        "game_id": snapshot.game_id,
        "takeover_turn": snapshot.turn,
        "outcome": outcome,
        "terminal_turn": int(state.time),
        "turns_after_takeover": int(state.time) - snapshot.turn,
    }


def batched_transition_equivalence(
    states: Any,
    *,
    learner_seat: int,
    opponent_kind: int,
    student_actions: np.ndarray,
    teacher_actions: np.ndarray,
    seed: int,
) -> np.ndarray:
    count = len(student_actions)
    opponent_seat = jnp.full((count,), 1 - learner_seat, dtype=jnp.int32)
    opponent_keys = jax.random.split(jax.random.PRNGKey(seed + 700_003), count)
    opponent_engine = batched_opponent_actions(
        states,
        opponent_seat,
        opponent_keys,
        tuple([opponent_kind] * count),
    )
    student_engine = jax.vmap(index_to_engine_action)(
        jnp.asarray(student_actions, dtype=jnp.int32)
    )
    teacher_engine = jax.vmap(index_to_engine_action)(
        jnp.asarray(teacher_actions, dtype=jnp.int32)
    )
    if learner_seat == 0:
        student_joint = jnp.stack([student_engine, opponent_engine], axis=1)
        teacher_joint = jnp.stack([teacher_engine, opponent_engine], axis=1)
    else:
        student_joint = jnp.stack([opponent_engine, student_engine], axis=1)
        teacher_joint = jnp.stack([opponent_engine, teacher_engine], axis=1)
    student_next = step_batch_jax(states, student_joint)
    teacher_next = step_batch_jax(states, teacher_joint)
    equal = np.ones((count,), dtype=bool)
    for left, right in zip(
        jax.tree_util.tree_leaves(student_next),
        jax.tree_util.tree_leaves(teacher_next),
        strict=True,
    ):
        left_np, right_np = np.asarray(left), np.asarray(right)
        axes = tuple(range(1, left_np.ndim))
        leaf_equal = left_np == right_np
        if axes:
            leaf_equal = np.all(leaf_equal, axis=axes)
        equal &= leaf_equal
    return equal


def process_captured_student_game(
    params: dict,
    batch: dict[str, Any],
    captured: tuple[Any, ...],
    job: dict[str, Any],
    index: int,
    rows: dict[str, list[Any]],
) -> tuple[dict[str, Any], list[Snapshot]]:
    (
        captured_states,
        captured_mem0,
        captured_mem1,
        _seats,
        episode_ids,
        rollout_keys,
    ) = captured
    same_episode = np.asarray(episode_ids[:, index]) == index
    done = np.asarray(batch["dones"][:, index], dtype=bool) & same_episode
    hits = np.flatnonzero(done)
    count = int(hits[0]) + 1 if len(hits) else int(np.count_nonzero(same_episode))
    states = jax.tree_util.tree_map(lambda value: value[:count, index], captured_states)
    spatial = np.asarray(batch["spatial"][:count, index])
    global_vec = np.asarray(batch["global"][:count, index])
    legal = np.asarray(batch["mask"][:count, index], dtype=bool)
    student_actions = np.asarray(batch["actions"][:count, index], dtype=np.int32)
    output = forward_batch(
        params,
        jnp.asarray(spatial, dtype=jnp.float32),
        jnp.asarray(global_vec, dtype=jnp.float32),
    )
    logits = np.asarray(output["flat_logits"])
    masked = np.where(legal, logits, -1.0e30)
    max_logits = np.max(masked, axis=1, keepdims=True)
    probabilities = np.exp(masked - max_logits)
    probabilities /= np.maximum(np.sum(probabilities, axis=1, keepdims=True), 1e-12)

    teacher = create_policy(TEACHER_ID)
    teacher_state = teacher.initial_state(GameContext(job["seat"], 21, 21))
    teacher_actions: list[int] = []
    snapshots: list[Snapshot] = []
    best_land = 0
    last_progress = 0
    stall_snapshotted = False
    first_divergence = -1
    for turn_index in range(count):
        state = jax.tree_util.tree_map(
            lambda value, i=turn_index: value[i], states
        )
        turn = int(state.time)
        memory_before = jax.tree_util.tree_map(
            lambda value, i=turn_index: value[i, index],
            captured_mem0 if job["seat"] == 0 else captured_mem1,
        )
        observation, _meta = engine_observation(state, job["seat"])
        land = int(batch["learner_land"][turn_index, index])
        if land > best_land:
            best_land = land
            last_progress = turn
        stalled = turn >= 128 and turn - last_progress >= 100
        if turn in CHECKPOINT_TURNS or (stalled and not stall_snapshotted):
            snapshots.append(
                Snapshot(
                    turn=turn,
                    state=clone_tree(state),
                    memory=clone_tree(memory_before),
                    teacher=copy.deepcopy(teacher),
                    teacher_state=copy.deepcopy(teacher_state),
                    opponent_kind=job["opponent_kind"],
                    rollout_key=jnp.array(rollout_keys[turn_index]),
                    opponent_env_index=index,
                    opponent_num_envs=len(batch["actions"][0]),
                    learner_seat=job["seat"],
                    game_id=job["game_id"],
                )
            )
            stall_snapshotted = stall_snapshotted or stalled
        teacher_action, teacher_state = policy_action(teacher, observation, teacher_state)
        teacher_index = action_to_index(teacher_action)
        if not legal[turn_index, teacher_index]:
            raise RuntimeError(
                f"illegal teacher label game={job['game_id']} turn={turn}"
            )
        teacher_actions.append(teacher_index)
        if teacher_index != int(student_actions[turn_index]) and first_divergence < 0:
            first_divergence = turn

    teacher_array = np.asarray(teacher_actions, dtype=np.int32)
    equivalent = batched_transition_equivalence(
        states,
        learner_seat=job["seat"],
        opponent_kind=job["opponent_kind"],
        student_actions=student_actions,
        teacher_actions=teacher_array,
        seed=job["seed"],
    )
    teacher_logits = logits[np.arange(count), teacher_array]
    teacher_rank = 1 + np.sum(legal & (logits > teacher_logits[:, None]), axis=1)
    exact = student_actions == teacher_array
    type_same = np.asarray(
        [
            action_kind(int(a)) == action_kind(int(b))
            for a, b in zip(student_actions, teacher_array, strict=True)
        ]
    )
    best_land = 0
    last_progress = 0
    for turn_index in range(count):
        turn = int(batch["turn"][turn_index, index])
        land = int(batch["learner_land"][turn_index, index])
        if land > best_land:
            best_land = land
            last_progress = turn
        stalled = turn >= 128 and turn - last_progress >= 100
        student_index = int(student_actions[turn_index])
        teacher_index = int(teacher_array[turn_index])
        confidence = float(probabilities[turn_index, student_index])
        high_information = (
            (not exact[turn_index] and not equivalent[turn_index])
            or stalled
            or (student_index == 0 and teacher_index != 0)
            or int(teacher_rank[turn_index]) > 8
            or confidence < 0.35
        )
        if not high_information:
            continue
        memory_after = ObsMemoryJax(
            seen_own=jnp.asarray(spatial[turn_index, 5], dtype=jnp.float32),
            last_army=jnp.asarray(spatial[turn_index, 6] * 100.0, dtype=jnp.float32),
        )
        append_row(
            rows,
            spatial=spatial[turn_index],
            global_vec=global_vec[turn_index],
            memory=memory_after,
            legal_mask=legal[turn_index],
            teacher_index=teacher_index,
            student_index=student_index,
            teacher_rank=int(teacher_rank[turn_index]),
            opponent=job["opponent"],
            seat=job["seat"],
            seed=job["seed"],
            turn=turn,
            game_id=job["game_id"],
            source="dagger_stall" if stalled else "dagger_disagreement",
            weight=2.5 if stalled else 2.0,
            divergence_turn=first_divergence,
            takeover_outcome="not_applicable",
        )
    if len(hits):
        terminal_index = int(hits[0])
        won = bool(batch["learner_won"][terminal_index, index])
        lost = bool(batch["learner_lost"][terminal_index, index])
        outcome = "win" if won else "loss" if lost else "draw"
        terminal_turn = int(batch["turn"][terminal_index, index]) + 1
    else:
        outcome = "draw"
        terminal_turn = TRUNCATION
    final = count - 1
    report = {
        "game_id": job["game_id"],
        "opponent": job["opponent"],
        "seat": job["seat"],
        "seed": job["seed"],
        "outcome": outcome,
        "terminal_turn": terminal_turn,
        "first_divergence": first_divergence,
        "exact_action_agreement": float(np.mean(exact)),
        "action_type_agreement": float(np.mean(type_same)),
        "transition_equivalent_disagreements": int(np.count_nonzero(equivalent & ~exact)),
        "learner_land": int(batch["learner_land"][final, index]),
        "opponent_land": int(batch["opponent_land"][final, index]),
        "learner_army": int(batch["learner_army"][final, index]),
        "opponent_army": int(batch["opponent_army"][final, index]),
        "snapshots": [snapshot.turn for snapshot in snapshots],
    }
    return report, snapshots


def teacher_path_snapshots(
    *, seed: int, opponent_name: str, learner_seat: int
) -> list[Snapshot]:
    game_id = f"teacher:{opponent_name}:seat{learner_seat}:seed{seed}"
    state = reset_one_jax(jax.random.PRNGKey(seed), 21, 21)
    teacher = create_policy(TEACHER_ID)
    teacher_state = teacher.initial_state(GameContext(learner_seat, 21, 21))
    opponent_kind = int(
        OpponentKind.PASS if opponent_name == "pass" else OpponentKind.RANDOM
    )
    rollout_key = jax.random.PRNGKey(seed + 100_003)
    memory = empty_memory()
    snapshots: list[Snapshot] = []
    for _ in range(TRUNCATION):
        turn = int(state.time)
        learner_obs, _ = engine_observation(state, learner_seat)
        if turn in (256, 512, 800):
            snapshots.append(
                Snapshot(
                    turn=turn,
                    state=clone_tree(state),
                    memory=clone_tree(memory),
                    teacher=copy.deepcopy(teacher),
                    teacher_state=copy.deepcopy(teacher_state),
                    opponent_kind=opponent_kind,
                    rollout_key=jnp.array(rollout_key),
                    opponent_env_index=0,
                    opponent_num_envs=1,
                    learner_seat=learner_seat,
                    game_id=game_id,
                )
            )
        teacher_action, teacher_state = policy_action(teacher, learner_obs, teacher_state)
        foe_action, rollout_key = opponent_action_from_rollout_key(
            state,
            learner_seat,
            opponent_kind,
            rollout_key,
            environment_index=0,
            num_environments=1,
        )
        _spatial, _global, memory = observe_one_jax(state, learner_seat, memory)
        state, info = transition_for_actions(
            state,
            learner_seat,
            _action_to_jax(teacher_action),
            foe_action,
        )
        if bool(info.is_done):
            break
    return snapshots


def trim_rows(rows: dict[str, list[Any]], maximum: int) -> None:
    count = len(rows["teacher_action"])
    if count <= maximum:
        return
    priority = {"takeover_recovery": 0, "dagger_stall": 1, "dagger_disagreement": 2}
    order = sorted(
        range(count),
        key=lambda index: (
            priority.get(str(rows["sample_source"][index]), 3),
            -int(rows["teacher_rank"][index]),
            str(rows["sample_id"][index]),
        ),
    )[:maximum]
    for key, values in list(rows.items()):
        if key != "_seen":
            rows[key] = [values[index] for index in order]


def save_dataset(rows: dict[str, list[Any]], output: Path, report: dict[str, Any]) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / "targeted_dagger_states.npz"
    np.savez_compressed(
        dataset,
        spatial=np.stack(rows["spatial"]),
        global_vec=np.stack(rows["global_vec"]),
        memory_seen_own=np.stack(rows["memory_seen_own"]),
        memory_last_army=np.stack(rows["memory_last_army"]),
        legal_mask=np.stack(rows["legal_mask"]),
        teacher_action=np.asarray(rows["teacher_action"], dtype=np.int32),
        student_action=np.asarray(rows["student_action"], dtype=np.int32),
        teacher_rank=np.asarray(rows["teacher_rank"], dtype=np.int16),
        opponent=np.asarray(rows["opponent"], dtype="U24"),
        seat=np.asarray(rows["seat"], dtype=np.int8),
        seed=np.asarray(rows["seed"], dtype=np.int32),
        turn=np.asarray(rows["turn"], dtype=np.int32),
        game_id=np.asarray(rows["game_id"], dtype="U96"),
        split=np.asarray(rows["split"], dtype="U16"),
        sample_id=np.asarray(rows["sample_id"], dtype="U32"),
        sample_source=np.asarray(rows["sample_source"], dtype="U24"),
        sample_weight=np.asarray(rows["sample_weight"], dtype=np.float32),
        phase=np.asarray(rows["phase"], dtype="U12"),
        divergence_turn=np.asarray(rows["divergence_turn"], dtype=np.int32),
        takeover_outcome=np.asarray(rows["takeover_outcome"], dtype="U16"),
    )
    report["dataset"] = str(dataset)
    report["dataset_sha256"] = sha256_file(dataset)
    report["samples"] = len(rows["teacher_action"])
    report["training_samples"] = int(np.count_nonzero(np.asarray(rows["split"]) == "train"))
    report["dagger_holdout_samples"] = int(
        np.count_nonzero(np.asarray(rows["split"]) == "dagger_holdout")
    )
    report["sample_sources"] = {
        name: rows["sample_source"].count(name)
        for name in sorted(set(rows["sample_source"]))
    }
    atomic_json(output / "dataset_manifest.json", report)
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--original-dataset", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--paired-seeds", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=120_000)
    parser.add_argument("--target-states", type=int, default=3_000)
    parser.add_argument("--minimum-states", type=int, default=1_500)
    parser.add_argument("--maximum-states", type=int, default=5_000)
    args = parser.parse_args()
    started = time.perf_counter()
    checkpoint_sha = sha256_file(args.checkpoint)
    original_sha = sha256_file(args.original_dataset)
    params = load_tree(args.checkpoint, init_params(jax.random.PRNGKey(0)))
    params = jax.device_put(params)
    rows = make_rows()
    student_games: list[dict[str, Any]] = []
    snapshots: list[Snapshot] = []
    captured_batch, captured_state, jobs = capture_student_rollouts(
        params,
        paired_seeds=args.paired_seeds,
        seed_base=args.seed_base,
    )
    for index, job in enumerate(jobs):
        game_report, game_snapshots = process_captured_student_game(
            params,
            captured_batch,
            captured_state,
            job,
            index,
            rows,
        )
        student_games.append(game_report)
        snapshots.extend(game_snapshots)
        print(
            f"CLOSED_LOOP game={game_report['game_id']} outcome={game_report['outcome']} "
            f"first_divergence={game_report['first_divergence']} "
            f"samples={len(rows['teacher_action'])}",
            flush=True,
        )
    if not snapshots:
        raise RuntimeError("no takeover snapshots collected")
    gate_snapshot = min(snapshots, key=lambda item: abs(item.turn - 64))
    shadow_clone_gate = clone_gate(gate_snapshot)
    takeover_results: list[dict[str, Any]] = []
    for snapshot in snapshots:
        opponent_name = "legal_random" if ":legal_random:" in snapshot.game_id else "pass"
        seed = int(snapshot.game_id.rsplit("seed", 1)[1])
        takeover_results.append(
            run_teacher_takeover(
                snapshot,
                params,
                rows,
                seed=seed,
                opponent_name=opponent_name,
            )
        )
    reverse_snapshots: list[Snapshot] = []
    for opponent_index, opponent_name in enumerate(OPPONENTS):
        seed = args.seed_base + opponent_index * 1_000
        for seat in (0, 1):
            reverse_snapshots.extend(
                teacher_path_snapshots(
                    seed=seed, opponent_name=opponent_name, learner_seat=seat
                )
            )
    reverse_results = [
        run_student_takeover(snapshot, params) for snapshot in reverse_snapshots
    ]
    trim_rows(rows, args.maximum_states)
    if len(rows["teacher_action"]) < args.minimum_states:
        raise RuntimeError(
            f"targeted DAgger minimum failed: {len(rows['teacher_action'])} < {args.minimum_states}"
        )
    later_teacher = [item for item in takeover_results if item["takeover_turn"] >= 256]
    teacher_later_win_rate = sum(item["outcome"] == "win" for item in later_teacher) / max(
        len(later_teacher), 1
    )
    reverse_later = [item for item in reverse_results if item["takeover_turn"] >= 256]
    student_teacher_state_win_rate = sum(
        item["outcome"] == "win" for item in reverse_later
    ) / max(len(reverse_later), 1)
    teacher_early = [item for item in takeover_results if item["takeover_turn"] <= 256]
    teacher_early_win_rate = sum(item["outcome"] == "win" for item in teacher_early) / max(
        len(teacher_early), 1
    )
    if teacher_later_win_rate >= 0.5 and student_teacher_state_win_rate < 0.5:
        diagnosis = "conversion_failure"
    elif teacher_early_win_rate < 0.5:
        diagnosis = "early_strategic_drift"
    else:
        diagnosis = "mixed_closed_loop_distribution_shift"
    identity = hashlib.sha256(
        json.dumps(
            {
                "checkpoint": checkpoint_sha,
                "original_dataset": original_sha,
                "seed_base": args.seed_base,
                "paired_seeds": args.paired_seeds,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    output = args.runtime / "dagger" / f"closed_loop_{identity[:16]}"
    report = {
        "schema_version": 1,
        "kind": "NOON_CLOSED_LOOP_DAGGER",
        "status": "PASS",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "original_dataset": str(args.original_dataset),
        "original_dataset_sha256": original_sha,
        "teacher_id": TEACHER_ID,
        "learner_state_contract": "stateless_policy_with_deterministic_observation_history",
        "TEACHER_SHADOW_CLONE_GATE": shadow_clone_gate,
        "student_games": student_games,
        "teacher_takeovers": takeover_results,
        "student_takeovers": reverse_results,
        "diagnosis": diagnosis,
        "teacher_later_takeover_win_rate": teacher_later_win_rate,
        "teacher_early_takeover_win_rate": teacher_early_win_rate,
        "student_from_teacher_state_win_rate": student_teacher_state_win_rate,
        "split_rule": "sha256(game_id) mod 5 == 0 is immutable dagger_holdout",
        "target_states": args.target_states,
        "minimum_states": args.minimum_states,
        "maximum_states": args.maximum_states,
        "elapsed_s": time.perf_counter() - started,
        "written_at": datetime.now(UTC).isoformat(),
    }
    dataset = save_dataset(rows, output, report)
    if report["training_samples"] == 0 or report["dagger_holdout_samples"] == 0:
        raise RuntimeError("deterministic DAgger train/holdout coverage failed")
    repo_report = {
        key: value
        for key, value in report.items()
        if key not in {"student_games", "teacher_takeovers", "student_takeovers"}
    }
    repo_report["dataset"] = str(dataset)
    repo_report["full_report"] = str(output / "dataset_manifest.json")
    atomic_json(
        ROOT / "experiments/manifests/noon_closed_loop_takeover.json", repo_report
    )
    print(json.dumps(repo_report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
