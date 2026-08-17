"""Shared serving for STAGE5_SCALE_R1 checkpoints (EVAL_ONLY).

Serves a registered SCALE terminal checkpoint (raw weights by default, EMA
via SCALE_WEIGHTS=ema) through the official stdio protocol using the
canonical 8-plane JaxTransformerPolicy (parity-proven inference lineage;
NO temporal history planes in this round per plan).
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
    JaxTransformerPolicy,
    load_jax_checkpoint,
)
from generals_bot.competition_native_jax.transformer_jax import init_params  # noqa: E402
from generals_bot.observation import GameContext  # noqa: E402
from generals_bot.policies.base import ActionDecision, PolicyState  # noqa: E402

CHECKPOINT_ROOTS = [
    _REPO / "experiments/marathon/screening_runs/STAGE5-SCALE-R2",
    _REPO / "experiments/marathon/screening_runs/STAGE5-SCALE-R1",
]


class _ScalePolicy:
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
    override = os.environ.get("SCALE_CKPT_DIR")
    npz = None
    if override:
        npz = Path(override) / f"{os.environ.get('SCALE_WEIGHTS', 'raw')}.npz"
    else:
        weights = os.environ.get("SCALE_WEIGHTS", "raw")
        for root in CHECKPOINT_ROOTS:
            candidate = root / arm_id / f"{weights}.npz"
            if candidate.is_file():
                npz = candidate
                break
    if npz is None or not npz.is_file():
        raise SystemExit(f"SCALE checkpoint not found for arm {arm_id}")
    template = init_params(jax.random.PRNGKey(0))
    params = load_jax_checkpoint(npz, template)
    run_agent(_ScalePolicy(JaxTransformerPolicy(params), f"scale_{arm_id.lower()}"),
              deterministic=True)
