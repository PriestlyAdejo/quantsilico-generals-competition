"""Legal action mask over the full 3970-dimensional support."""

from __future__ import annotations

import numpy as np

from generals_bot.action import Action
from generals_bot.castle_cost import castle_price_at, own_structures
from generals_bot.competition_native_jax.action_codec import action_to_index
from generals_bot.competition_native_jax.constants import ACTION_DIM, MAX_HW, PASS_INDEX
from generals_bot.legal import enumerate_legal_actions
from generals_bot.observation import Observation


def legal_mask_from_observation(observation: Observation) -> np.ndarray:
    """Boolean mask length ACTION_DIM; True = legal.

    Uses enumerate_legal_actions for playable HxW cells. Padding cells remain
    illegal. PASS is always legal.
    """
    mask = np.zeros((ACTION_DIM,), dtype=bool)
    mask[PASS_INDEX] = True
    for action in enumerate_legal_actions(observation):
        # Skip actions that reference cells outside padded board (should not happen)
        if action.row >= MAX_HW or action.col >= MAX_HW:
            continue
        mask[action_to_index(action)] = True
    return mask


def masked_log_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Log-softmax over legal support only; illegal positions -> -inf."""
    logits = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("empty legal mask")
    x = np.where(mask, logits, -1e30)
    x = x - np.max(x[mask])
    exp = np.where(mask, np.exp(x), 0.0)
    z = exp.sum()
    return np.where(mask, x - np.log(z), -np.inf)


def sample_action(logits: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[Action, float]:
    from generals_bot.competition_native_jax.action_codec import index_to_action

    logp = masked_log_softmax(logits, mask)
    probs = np.where(mask, np.exp(logp), 0.0)
    idx = int(rng.choice(ACTION_DIM, p=probs / probs.sum()))
    return index_to_action(idx), float(logp[idx])


def greedy_action(logits: np.ndarray, mask: np.ndarray) -> Action:
    from generals_bot.competition_native_jax.action_codec import index_to_action

    logp = masked_log_softmax(logits, mask)
    return index_to_action(int(np.argmax(np.where(mask, logp, -np.inf))))
