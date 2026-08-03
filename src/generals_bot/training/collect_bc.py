"""In-process heuristic trajectory collection for behaviour cloning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch

from generals import GeneralsEnv
from generals.core import game

from generals_bot.evaluation.match import make_board, make_transition
from generals_bot.models.action_index import ACTION_DIM, action_to_index
from generals_bot.models.heads import STRATEGIC_OPTIONS
from generals_bot.models.legal_mask import legal_mask_observation
from generals_bot.models.observation_encoder import encode_globals_numpy, encode_grids_numpy
from generals_bot.observation import GameContext, Observation
from generals_bot.policies.base import TraceLevel
from generals_bot.selector import create_policy
from generals_bot.training.bridge_benchmark import extract_numpy_boards

OPTION_TO_IDX = {name: i for i, name in enumerate(STRATEGIC_OPTIONS)}


@dataclass
class BCSample:
    cells: np.ndarray
    globals_: np.ndarray
    action_index: int
    option_index: int
    legal_mask: np.ndarray
    source: str


def _observation_from_arrays(
    type_grid: np.ndarray,
    owner: np.ndarray,
    armies: np.ndarray,
    meta: dict,
) -> Observation:
    return Observation(
        height=int(type_grid.shape[0]),
        width=int(type_grid.shape[1]),
        turn=int(meta["turn"]),
        my_land=int(meta["my_land"]),
        my_army=int(meta["my_army"]),
        opp_land=int(meta["opp_land"]),
        opp_army=int(meta["opp_army"]),
        type_grid=tuple(tuple(int(x) for x in row) for row in type_grid),
        owner_grid=tuple(tuple(int(x) for x in row) for row in owner),
        army_grid=tuple(tuple(int(x) for x in row) for row in armies),
    )


def _action_to_jax(action) -> jnp.ndarray:
    return jnp.array(
        [action.kind, action.row, action.col, action.direction, action.split],
        dtype=jnp.int32,
    )


def collect_trajectories(
    *,
    policies: list[str],
    seeds: list[int],
    max_turns: int = 80,
    opponent: str = "pass",
) -> list[BCSample]:
    env = GeneralsEnv(mode="competition")
    transition = make_transition(env)
    get_obs = game.get_observation
    samples: list[BCSample] = []

    for seed in seeds:
        for policy_name in policies:
            state = make_board(env, seed)
            h, w = (int(d) for d in state.armies.shape)
            learner = create_policy(policy_name)
            foe = create_policy(opponent, seed=seed + 17)
            ctx0 = GameContext(player_id=0, height=h, width=w)
            ctx1 = GameContext(player_id=1, height=h, width=w)
            st0 = learner.initial_state(ctx0)
            st1 = foe.initial_state(ctx1)

            for _turn in range(max_turns):
                eng0 = get_obs(state, 0)
                eng1 = get_obs(state, 1)
                t0, o0, a0, _g0, meta0 = extract_numpy_boards(eng0, h, w)
                t1, o1, a1, _g1, meta1 = extract_numpy_boards(eng1, h, w)
                obs0 = _observation_from_arrays(t0, o0, a0, meta0)
                obs1 = _observation_from_arrays(t1, o1, a1, meta1)

                dec0 = learner.act(
                    obs0, st0, deterministic=True, trace=TraceLevel.NONE, deadline=None
                )
                dec1 = foe.act(
                    obs1, st1, deterministic=True, trace=TraceLevel.NONE, deadline=None
                )
                st0 = dec0.new_state
                st1 = dec1.new_state

                cells = encode_grids_numpy(t0, o0, a0)
                glob = encode_globals_numpy(obs0)
                opt = OPTION_TO_IDX.get(dec0.strategic_option, OPTION_TO_IDX["WAIT"])
                mask = legal_mask_observation(obs0).cpu().numpy().astype(bool)
                assert mask[action_to_index(dec0.action)]
                samples.append(
                    BCSample(
                        cells=cells,
                        globals_=glob,
                        action_index=action_to_index(dec0.action),
                        option_index=opt,
                        legal_mask=mask,
                        source=policy_name,
                    )
                )

                actions = jnp.stack([_action_to_jax(dec0.action), _action_to_jax(dec1.action)])
                state, info = transition(state, actions)
                if bool(info.is_done):
                    break
    return samples


def save_dataset(samples: list[BCSample], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cells": np.stack([s.cells for s in samples]),
        "globals": np.stack([s.globals_ for s in samples]),
        "action_index": np.asarray([s.action_index for s in samples], dtype=np.int64),
        "option_index": np.asarray([s.option_index for s in samples], dtype=np.int64),
        "legal_mask": np.stack([s.legal_mask for s in samples]).astype(bool),
        "source": np.asarray([s.source for s in samples]),
    }
    np.savez_compressed(path, **payload)
    meta = {
        "n": len(samples),
        "sources": sorted({s.source for s in samples}),
        "path": str(path),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path


def load_dataset(path: Path) -> dict[str, torch.Tensor]:
    data = np.load(path, allow_pickle=True)
    return {
        "cells": torch.from_numpy(data["cells"].astype(np.float32)),
        "globals": torch.from_numpy(data["globals"].astype(np.float32)),
        "action_index": torch.from_numpy(data["action_index"].astype(np.int64)),
        "option_index": torch.from_numpy(data["option_index"].astype(np.int64)),
        "legal_mask": torch.from_numpy(data["legal_mask"].astype(bool)),
    }
