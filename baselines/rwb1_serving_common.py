"""Shared serving for REWARD-BRIDGE-R1 checkpoints (EVAL_ONLY).

Serves a registered RWB1 terminal checkpoint (raw weights by default, EMA via
RWB1_WEIGHTS=ema) through the official stdio protocol, mirroring the
marathon_baseline_v0 serving path (parity-proven inference, protocol adapter).
PPO_SEMANTICS: EVAL_ONLY - never touches training action selection.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_BASELINES = Path(__file__).resolve().parent
_REPO = _BASELINES.parent
_SRC = _REPO / "src"
for entry in (_SRC, _REPO):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import jax  # noqa: E402

from generals_bot.agent import run_agent  # noqa: E402
from generals_bot.competition_native_jax.jax_baseline_policy import (  # noqa: E402
    JaxTransformerPolicy,
    load_jax_checkpoint,
)
from generals_bot.competition_native_jax.transformer_jax import init_params  # noqa: E402
from generals_bot.observation import GameContext  # noqa: E402
from generals_bot.policies.base import ActionDecision, PolicyState  # noqa: E402

CHECKPOINT_ROOT = _REPO / "experiments/marathon/screening_runs_reward_bridge_r1"


class _Rwb1Policy:
    def __init__(self, inner: JaxTransformerPolicy, policy_id: str) -> None:
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
    ckpt_dir = Path(os.environ.get("RWB1_CKPT_DIR", str(CHECKPOINT_ROOT / arm_id)))
    weights = os.environ.get("RWB1_WEIGHTS", "raw")
    npz = ckpt_dir / f"{weights}.npz"
    if not npz.is_file():
        raise SystemExit(f"RWB1 checkpoint not found: {npz}")
    template = init_params(jax.random.PRNGKey(0))
    params = load_jax_checkpoint(npz, template)
    run_agent(_Rwb1Policy(JaxTransformerPolicy(params), f"rwb1_{arm_id.lower()}"),
              deterministic=True)
