"""Deployable NumPy student policy (emb96/d2/h4) for competition stdio ZIP.

Recurrent state for this architecture is ObsMemory (seen_own / last_army / turn),
not an RNN hidden vector inside the transformer. Sequence contract:
  - zero/reset ObsMemory only at true episode boundaries
  - fragment boundaries carry ObsMemory through burn-in by replaying context steps
  - padding steps must not update ObsMemory
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from generals_bot.competition_native_jax.constants import (
    HL_GAUSS_BINS,
    NUM_PATCHES,
    PATCH,
)
from generals_bot.competition_native_jax.obs_memory import N_GLOBAL, N_SPATIAL, ObsMemory, encode_observation
from generals_bot.competition_native_jax.policy import CompetitionNativePolicy, load_weights, save_weights
from generals_bot.competition_native_jax.transformer import TransformerWeights, forward, weights_to_dict
from generals_bot.observation import Observation

STUDENT_EMB = 96
STUDENT_LAYERS = 2
STUDENT_HEADS = 4


def init_student_weights(
    seed: int = 0,
    *,
    emb: int = STUDENT_EMB,
    layers: int = STUDENT_LAYERS,
    heads: int = STUDENT_HEADS,
) -> TransformerWeights:
    if emb % heads != 0:
        raise ValueError(f"emb {emb} not divisible by heads {heads}")
    rng = np.random.default_rng(seed)

    def xavier(shape: tuple[int, ...]) -> np.ndarray:
        fan_in = shape[0] if len(shape) >= 2 else shape[-1]
        fan_out = shape[-1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=shape).astype(np.float32)

    w = TransformerWeights(
        patch_proj=xavier((N_SPATIAL * PATCH * PATCH, emb)),
        cls=xavier((emb,)),
        pos=xavier((NUM_PATCHES + 1, emb)),
        global_proj=xavier((N_GLOBAL, emb)),
        attn_w=[],
        attn_out=[],
        ff_w1=[],
        ff_w2=[],
        move_head=xavier((emb, PATCH * PATCH * 8)),
        build_head=xavier((emb, PATCH * PATCH)),
        pass_head=xavier((emb,)),
        value_head=xavier((emb, HL_GAUSS_BINS)),
    )
    for _ in range(layers):
        w.attn_w.append(xavier((emb, 3 * emb)))
        w.attn_out.append(xavier((emb, emb)))
        w.ff_w1.append(xavier((emb, 4 * emb)))
        w.ff_w2.append(xavier((4 * emb, emb)))
    return w


def validate_student_weights(weights: TransformerWeights, *, emb: int = STUDENT_EMB, layers: int = STUDENT_LAYERS) -> None:
    if int(weights.patch_proj.shape[1]) != emb:
        raise ValueError(f"expected emb={emb}, got {weights.patch_proj.shape[1]}")
    if len(weights.attn_w) != layers:
        raise ValueError(f"expected layers={layers}, got {len(weights.attn_w)}")


def student_architecture_hash(*, emb: int = STUDENT_EMB, layers: int = STUDENT_LAYERS, heads: int = STUDENT_HEADS) -> str:
    blob = f"student_emb{emb}_d{layers}_h{heads}|N_SPATIAL={N_SPATIAL}|N_GLOBAL={N_GLOBAL}|PATCH={PATCH}".encode()
    return hashlib.sha256(blob).hexdigest()


def observation_schema_hash() -> str:
    # Channel order in encode_observation: playable, me, opp, army/100, mountain, seen_own, last_army/100, prices
    blob = b"spatial[8]=playable,me,opp,army100,mountain,seen_own,last_army100,prices|global[8]=hz*5,h/max,w/max"
    return hashlib.sha256(blob).hexdigest()


class StudentCompetitionNativePolicy(CompetitionNativePolicy):
    """NumPy deploy policy with explicit student shape validation."""

    def __init__(self, weights: TransformerWeights | None = None, seed: int = 0) -> None:
        if weights is None:
            weights = init_student_weights(seed)
        validate_student_weights(weights)
        super().__init__(weights=weights, seed=seed)

    def memory_snapshot(self) -> dict[str, Any]:
        return {
            "seen_own": self.memory.seen_own.copy(),
            "last_army": self.memory.last_army.copy(),
            "turn": int(self.memory.turn),
            "height": int(self.memory.height),
            "width": int(self.memory.width),
        }

    def restore_memory(self, snap: dict[str, Any]) -> None:
        self.memory.seen_own = np.asarray(snap["seen_own"], dtype=np.float32)
        self.memory.last_army = np.asarray(snap["last_army"], dtype=np.float32)
        self.memory.turn = int(snap["turn"])
        self.memory.height = int(snap["height"])
        self.memory.width = int(snap["width"])


def unroll_sequence_numpy(
    policy: StudentCompetitionNativePolicy,
    observations: list[Observation],
    *,
    valid_mask: np.ndarray,
    episode_reset: np.ndarray,
) -> dict[str, np.ndarray]:
    """Unroll ObsMemory recurrent contract over a sequence.

    valid_mask[t]=1 → update memory + forward; =0 → carry memory, skip loss position.
    episode_reset[t]=1 → reset memory before step t (true episode boundary).
    """
    T = len(observations)
    logits = []
    values = []
    for t in range(T):
        if bool(episode_reset[t]):
            h = int(observations[t].height)
            w = int(observations[t].width)
            policy.reset(h, w)
        if not bool(valid_mask[t]):
            # padding: do not update memory; emit zeros (masked from loss)
            logits.append(np.zeros((3970,), dtype=np.float32))
            values.append(np.zeros((HL_GAUSS_BINS,), dtype=np.float32))
            continue
        spatial, g = encode_observation(observations[t], policy.memory)
        out = forward(spatial, g, policy.weights)
        logits.append(out["flat_logits"].astype(np.float32))
        values.append(out["value_logits"].astype(np.float32))
    return {"flat_logits": np.stack(logits, axis=0), "value_logits": np.stack(values, axis=0)}


def schema_hashes() -> dict[str, str]:
    return {
        "observation_schema_hash": observation_schema_hash(),
        "normalisation_hash": hashlib.sha256(b"army/100,prices/100,clip2,playable_pad_mountain").hexdigest(),
        "action_mapping_hash": hashlib.sha256(b"PASS+9*cell_move8_build1_ACTION_DIM3970").hexdigest(),
        "legal_support_hash": hashlib.sha256(b"legal_mask_from_observation").hexdigest(),
        "student_architecture_hash": student_architecture_hash(),
    }
