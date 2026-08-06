"""Weights IO and policy wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from generals_bot.action import Action
from generals_bot.competition_native_jax.legal_mask import greedy_action, legal_mask_from_observation, sample_action
from generals_bot.competition_native_jax.obs_memory import ObsMemory, encode_observation
from generals_bot.competition_native_jax.transformer import (
    TransformerWeights,
    forward,
    init_weights,
    weights_from_dict,
    weights_to_dict,
)
from generals_bot.observation import Observation


def save_weights(path: Path, weights: TransformerWeights) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **weights_to_dict(weights))


def load_weights(path: Path) -> TransformerWeights:
    data = np.load(path)
    return weights_from_dict({k: data[k] for k in data.files})


class CompetitionNativePolicy:
    """CPU NumPy policy for deployment and training prototypes."""

    def __init__(self, weights: TransformerWeights | None = None, seed: int = 0) -> None:
        self.weights = weights if weights is not None else init_weights(seed)
        self.memory = ObsMemory()
        self.rng = np.random.default_rng(seed)

    def reset(self, height: int, width: int) -> None:
        self.memory.reset(height, width)

    def act(
        self,
        observation: Observation,
        *,
        deterministic: bool = True,
    ) -> tuple[Action, dict]:
        spatial, global_vec = encode_observation(observation, self.memory)
        out = forward(spatial, global_vec, self.weights)
        mask = legal_mask_from_observation(observation)
        if deterministic:
            action = greedy_action(out["flat_logits"], mask)
            logp = None
        else:
            action, logp = sample_action(out["flat_logits"], mask, self.rng)
        return action, {"mask": mask, "logits": out["flat_logits"], "logp": logp, "value_logits": out["value_logits"]}
