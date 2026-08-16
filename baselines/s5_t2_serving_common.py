"""Shared serving for STAGE5 T2 K1 checkpoints (EVAL_ONLY).

Serves a registered T2 terminal checkpoint (raw weights by default, EMA via
S5T2_WEIGHTS=ema) through the official stdio protocol using the k1-aware
JaxTransformerHistoryPolicy adapter (prev-tick legal spatial planes 8-16,
zero at game start - matching the training rollout convention). The shared
inference path is parity-proven (EV-0019/0034 lineage).
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
    JaxTransformerHistoryPolicy,
    load_jax_checkpoint,
)
from generals_bot.competition_native_jax.transformer_jax import init_params  # noqa: E402
from generals_bot.observation import GameContext  # noqa: E402
from generals_bot.policies.base import ActionDecision, PolicyState  # noqa: E402

CHECKPOINT_ROOT = _REPO / "experiments/marathon/screening_runs/STAGE5-T2-R1/artefacts"
K1_SPATIAL_PLANES = 16


class _S5T2Policy:
    def __init__(self, inner: JaxTransformerHistoryPolicy, policy_id: str) -> None:
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
    ckpt_dir = Path(os.environ.get("S5T2_CKPT_DIR", str(CHECKPOINT_ROOT / arm_id)))
    weights = os.environ.get("S5T2_WEIGHTS", "raw")
    npz = ckpt_dir / f"{weights}.npz"
    if not npz.is_file():
        raise SystemExit(f"T2 checkpoint not found: {npz}")
    template = init_params(jax.random.PRNGKey(0), spatial_planes=K1_SPATIAL_PLANES)
    params = load_jax_checkpoint(npz, template)
    run_agent(
        _S5T2Policy(JaxTransformerHistoryPolicy(params), f"s5_t2_{arm_id.lower()}"),
        deterministic=True,
    )
