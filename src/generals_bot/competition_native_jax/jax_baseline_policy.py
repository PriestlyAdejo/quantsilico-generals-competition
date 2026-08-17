"""JAX-lineage transformer policy served over the stdio protocol (EVAL_ONLY).

PPO_SEMANTICS: EVAL_ONLY. This module never participates in training action
selection; it serves frozen checkpoint weights through the exact serving path
whose parity with the training observation pipeline is proven by
tests/unit/test_baseline_agent_parity.py.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from generals_bot.action import Action
from generals_bot.competition_native_jax.action_codec import index_to_action
from generals_bot.competition_native_jax.inference_jax import infer
from generals_bot.competition_native_jax.legal_mask import legal_mask_from_observation
from generals_bot.competition_native_jax.obs_memory import (
    ObsMemory,
    ObsMemoryV2,
    encode_observation,
    encode_observation_v2,
)
from generals_bot.observation import Observation
from generals_bot.policies.base import ActionDecision, PolicyState


@jax.jit
def _jit_infer(params, spatial, global_vec, mask, key=None):
    return infer(params, spatial, global_vec, mask, key)


class JaxTransformerPolicy:
    """Frozen JAX transformer checkpoint served deterministically."""

    policy_id = "marathon_jax_baseline"

    def __init__(self, params: dict, *, seed: int = 0) -> None:
        self.params = params
        self.memory = ObsMemory()
        self.seed = seed
        self._key = jax.random.PRNGKey(seed)

    def reset(self, height: int, width: int) -> None:
        self.memory.reset(height, width)

    def act(self, observation: Observation, *, deterministic: bool = True) -> Action:
        spatial, global_vec = encode_observation(observation, self.memory)
        mask = legal_mask_from_observation(observation)
        key = None
        if not deterministic:
            self._key = jax.random.fold_in(self._key, int(observation.turn))
            key = self._key
        idx, _logp, _out = _jit_infer(
            self.params,
            jnp.asarray(spatial),
            jnp.asarray(global_vec),
            jnp.asarray(mask),
            key,
        )
        return index_to_action(int(np.asarray(idx)))


def load_jax_checkpoint(raw_npz: Path, template: dict) -> dict:
    """Load raw weights via the canonical schema-v2 tree loader lineage."""
    from train.competition_native_jax.train_jax import load_tree

    return load_tree(raw_npz, template)


def load_params_npz(raw_npz: Path, template: dict) -> dict:
    """Self-contained npz -> params loader for submission packages.

    Byte-compatible with train_jax.load_tree (same save_tree key layout) but
    imports nothing outside generals_bot, so it works inside a packaged ZIP.
    """
    data = np.load(raw_npz, allow_pickle=False)
    flat_like, treedef = jax.tree_util.tree_flatten_with_path(template)
    leaves = []
    for key_path, leaf in flat_like:
        arr = data[str(key_path)]
        leaves.append(jnp.asarray(arr, dtype=getattr(leaf, "dtype", arr.dtype)))
    return jax.tree_util.tree_unflatten(treedef, leaves)


class JaxTransformerHistoryPolicy(JaxTransformerPolicy):
    """STAGE5 T2 serving variant: k1 legal temporal history (EVAL_ONLY).

    Appends the previous tick's LEGAL spatial observation as planes 8-16,
    exactly matching the training rollout convention (zero history at game
    start and after every episode boundary). Only the observation width seen
    by the frozen weights changes; action selection semantics are unchanged.
    """

    policy_id = "marathon_jax_t2_history"

    def reset(self, height: int, width: int) -> None:
        super().reset(height, width)
        from generals_bot.competition_native_jax.constants import MAX_HW
        from generals_bot.competition_native_jax.obs_memory import N_SPATIAL

        self._prev_spatial = np.zeros((N_SPATIAL, MAX_HW, MAX_HW), dtype=np.float32)

    def act(self, observation: Observation, *, deterministic: bool = True) -> Action:
        spatial, global_vec = encode_observation(observation, self.memory)
        spatial_hist = np.concatenate([spatial, self._prev_spatial], axis=0)
        self._prev_spatial = np.asarray(spatial, dtype=np.float32)
        mask = legal_mask_from_observation(observation)
        key = None
        if not deterministic:
            self._key = jax.random.fold_in(self._key, int(observation.turn))
            key = self._key
        idx, _logp, _out = _jit_infer(
            self.params,
            jnp.asarray(spatial_hist),
            jnp.asarray(global_vec),
            jnp.asarray(mask),
            key,
        )
        return index_to_action(int(np.asarray(idx)))


class JaxTransformerProtocolPolicy:
    """Protocol adapter so the frozen JAX policy can run under run_agent.

    PPO_SEMANTICS: EVAL_ONLY. JaxTransformerPolicy speaks reset()/act();
    the stdio competition loop requires the Policy protocol
    (initial_state/act -> ActionDecision). This adapter bridges the two
    without touching the parity-proven inference path.
    """

    policy_id = "marathon_jax_baseline_protocol"

    def __init__(self, params: dict, *, seed: int = 0) -> None:
        self._inner = JaxTransformerPolicy(params, seed=seed)

    def initial_state(self, context) -> PolicyState:
        self._inner.reset(context.height, context.width)
        return PolicyState()

    def act(
        self,
        observation: Observation,
        state: PolicyState,
        *,
        deterministic: bool,
        trace=None,
        deadline: float | None = None,
    ) -> ActionDecision:
        action = self._inner.act(observation, deterministic=deterministic)
        return ActionDecision(
            action=action,
            new_state=state,
            policy_id=self.policy_id,
        )


class JaxTransformerObsV2Policy:
    """OBS_V2_R1 serving variant: objective-aware 14-plane/12-global observation
    (EVAL_ONLY).

    Uses encode_observation_v2 + ObsMemoryV2, exactly matching the training
    rollout convention (obs_v2_jax); training/serving parity is proven by
    tests/competition_native_jax/test_obs_v2_parity.py. Only the observation
    encoding changes; action selection semantics are unchanged.
    """

    policy_id = "marathon_jax_obs_v2"

    def __init__(self, params: dict, *, seed: int = 0) -> None:
        self.params = params
        self.memory = ObsMemoryV2()
        self.seed = seed
        self._key = jax.random.PRNGKey(seed)

    def reset(self, height: int, width: int) -> None:
        self.memory.reset(height, width)

    def act(self, observation: Observation, *, deterministic: bool = True) -> Action:
        spatial, global_vec = encode_observation_v2(observation, self.memory)
        mask = legal_mask_from_observation(observation)
        key = None
        if not deterministic:
            self._key = jax.random.fold_in(self._key, int(observation.turn))
            key = self._key
        idx, _logp, _out = _jit_infer(
            self.params,
            jnp.asarray(spatial),
            jnp.asarray(global_vec),
            jnp.asarray(mask),
            key,
        )
        return index_to_action(int(np.asarray(idx)))
