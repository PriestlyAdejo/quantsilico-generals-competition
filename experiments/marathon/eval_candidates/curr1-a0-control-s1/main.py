"""CURRICULUM-R1 control opponent: A0-CONTROL seed S1 terminal checkpoint.

PPO_SEMANTICS: EVAL_ONLY. Serves frozen training weights deterministically
through the parity-proven JAX serving path (EV-0019, EV-0034 adapter).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import jax  # noqa: E402

from generals_bot.agent import run_agent  # noqa: E402
from generals_bot.competition_native_jax.jax_baseline_policy import (  # noqa: E402
    JaxTransformerProtocolPolicy,
)
from generals_bot.competition_native_jax.transformer_jax import init_params  # noqa: E402
from train.competition_native_jax.train_jax import load_tree  # noqa: E402

CHECKPOINT = Path(__file__).resolve().parent / "weights" / "raw.npz"


def main() -> None:
    params_like = init_params(jax.random.PRNGKey(0))
    params = load_tree(CHECKPOINT, params_like)
    run_agent(JaxTransformerProtocolPolicy(params), deterministic=True)


if __name__ == "__main__":
    main()
