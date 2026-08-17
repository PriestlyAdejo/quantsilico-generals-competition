"""Shared serving for OBS_V2_R1 terminal checkpoints (EVAL_ONLY).

Serves a registered OBS-V2-R1 arm terminal checkpoint (raw weights by
default, EMA via OBSV2_WEIGHTS=ema) through the official stdio protocol
using the OBS-V2 14-plane/12-global JaxTransformerObsV2Policy (parity with
the training path proven by tests/competition_native_jax/test_obs_v2_parity.py).
PPO_SEMANTICS: EVAL_ONLY - never touches training action selection.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_BASELINES = Path(__file__).parent
_REPO = _BASELINES.parent
_SRC = _REPO / "src"
for entry in (_SRC, _REPO):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import jax  # noqa: E402

from generals_bot.agent import run_agent  # noqa: E402
from generals_bot.competition_native_jax.jax_baseline_policy import (  # noqa: E402
    JaxTransformerObsV2Policy,
    load_jax_checkpoint,
)
from generals_bot.competition_native_jax.obs_v2_jax import (  # noqa: E402
    N_GLOBAL_V2,
    N_SPATIAL_V2,
)
from generals_bot.competition_native_jax.transformer_jax import init_params  # noqa: E402
from generals_bot.observation import GameContext  # noqa: E402
from generals_bot.policies.base import ActionDecision, PolicyState  # noqa: E402

CHECKPOINT_ROOTS = [
    _REPO / "experiments/marathon/obs_v2_r1",
]


class _ObsV2Policy:
    def __init__(self, inner: JaxTransformerObsV2Policy, policy_id: str) -> None:
        self.inner = inner
        self.policy_id = policy_id

    def initial_state(self, context: GameContext) -> PolicyState:
        self.inner.reset(context.height, context.width)
        return PolicyState(data={"player_id": context.player_id})

    def act(self, observation, state, *, deterministic, trace, deadline):
        action = self.inner.act(observation, deterministic=deterministic)
        return ActionDecision(
            action=action,
            new_state=state,
            policy_id=self.policy_id,
            strategic_option="LEARNED",
        )


def serve(arm_id: str) -> None:
    override = os.environ.get("OBSV2_CKPT_DIR")
    npz = None
    if override:
        npz = Path(override) / f"{os.environ.get('OBSV2_WEIGHTS', 'raw')}.npz"
    else:
        weights = os.environ.get("OBSV2_WEIGHTS", "raw")
        for root in CHECKPOINT_ROOTS:
            candidate = root / arm_id / f"{weights}.npz"
            if candidate.is_file():
                npz = candidate
                break
    if npz is None or not npz.is_file():
        raise SystemExit(f"OBS-V2 checkpoint not found for arm {arm_id}")
    template = init_params(
        jax.random.PRNGKey(0),
        spatial_planes=N_SPATIAL_V2,
        global_dim=N_GLOBAL_V2,
    )
    params = load_jax_checkpoint(npz, template)
    run_agent(_ObsV2Policy(JaxTransformerObsV2Policy(params), f"obsv2_{arm_id.lower()}"),
              deterministic=True)
