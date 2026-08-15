"""MARATHON_BASELINE_V0 protocol agent (EVAL_ONLY).

Serves the frozen SPRINT_VALID_PPO_7M59 checkpoint (raw weights) through the
official stdio protocol. PPO_SEMANTICS: EVAL_ONLY — this agent never touches
training action selection. Observation parity with the training pipeline is
proven by tests/unit/test_baseline_agent_parity.py; checkpoint identity is
hash-verified (EV-0002/0010/0013/0016).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
_REPO = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import jax  # noqa: E402

from generals_bot.agent import run_agent  # noqa: E402
from generals_bot.competition_native_jax.jax_baseline_policy import (  # noqa: E402
    JaxTransformerPolicy,
    load_jax_checkpoint,
)
from generals_bot.competition_native_jax.transformer_jax import init_params  # noqa: E402
from generals_bot.observation import GameContext  # noqa: E402
from generals_bot.policies.base import ActionDecision, PolicyState  # noqa: E402

DEFAULT_CHECKPOINT = (
    Path.home()
    / "quantsilico-runtime"
    / "cloud_assisted_deadline_salvage_v1_final"
    / "ckpt_final_u482_t7593984"
)


class _MarathonBaselinePolicy:
    policy_id = "marathon_baseline_v0"

    def __init__(self, inner: JaxTransformerPolicy) -> None:
        self.inner = inner

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


def main() -> None:
    checkpoint = Path(
        os.environ.get("MARATHON_BASELINE_V0_CKPT", str(DEFAULT_CHECKPOINT))
    )
    raw_npz = checkpoint / "raw.npz"
    if not raw_npz.is_file():
        raise SystemExit(f"baseline checkpoint not found: {raw_npz}")
    template = init_params(jax.random.PRNGKey(0))
    params = load_jax_checkpoint(raw_npz, template)
    run_agent(_MarathonBaselinePolicy(JaxTransformerPolicy(params)), deterministic=True)


if __name__ == "__main__":
    main()
