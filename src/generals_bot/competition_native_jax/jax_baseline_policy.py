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
from generals_bot.competition_native_jax.obs_memory import ObsMemory, encode_observation
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
