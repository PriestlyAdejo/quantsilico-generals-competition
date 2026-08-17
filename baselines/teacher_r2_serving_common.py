"""Shared serving for the STAGE5_TEACHER_R2 BC checkpoint (EVAL_ONLY).

Serves the hunter-self-play distilled BC checkpoint through the official
stdio protocol. Observation encoding uses the canonical inference path
(obs_memory.encode_observation with per-game fog memory) and action
selection is masked argmax over the enumerated legal actions using the
BC-A canonical 3970-index action layout (engine_action_to_index).
PPO_SEMANTICS: EVAL_ONLY - never touches training action selection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

_BASELINES = Path(__file__).parent
_REPO = _BASELINES.parent
for entry in (_REPO / "src", _REPO, _REPO / "third_party" / "generals-bots"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from generals_bot.agent import run_agent  # noqa: E402
from generals_bot.competition_native_jax.obs_memory import (  # noqa: E402
    ObsMemory,
    encode_observation,
)
from generals_bot.legal import enumerate_legal_actions  # noqa: E402
from generals_bot.observation import GameContext  # noqa: E402
from generals_bot.policies.base import ActionDecision, PolicyState  # noqa: E402
from scripts.training.bc_a_train_pilot import init_params, small_policy  # noqa: E402

PARAMS_NPZ = _REPO / "experiments/marathon/teacher_r2/step3_bc/params.npz"
MAX_HW = 21


def engine_action_to_index(action) -> int:
    kind, row, col, direction, split = action
    if kind == 1:
        return 0
    cell = row * MAX_HW + col
    if kind == 2:
        return 1 + cell * 9 + 8
    return 1 + cell * 9 + direction * 2 + split


def load_params() -> dict:
    data = np.load(PARAMS_NPZ)
    return {
        "conv_w": [jnp.asarray(data["conv_w_0"]), jnp.asarray(data["conv_w_1"])],
        "conv_b": [jnp.asarray(data["conv_b_0"]), jnp.asarray(data["conv_b_1"])],
        "fc_w": jnp.asarray(data["fc_w"]),
        "fc_b": jnp.asarray(data["fc_b"]),
        "out_w": jnp.asarray(data["out_w"]),
        "out_b": jnp.asarray(data["out_b"]),
    }


class _TeacherBcPolicy:
    policy_id = "teacher_r2_bc_s1"

    def __init__(self, params: dict) -> None:
        self.params = params
        self.memory = ObsMemory()

    def initial_state(self, context: GameContext) -> PolicyState:
        self.memory.reset(context.height, context.width)
        return PolicyState(data={"player_id": context.player_id})

    def act(self, observation, state, *, deterministic, trace, deadline):
        spatial, global_vec = encode_observation(observation, self.memory)
        logits = small_policy(
            self.params,
            jnp.asarray(spatial, dtype=jnp.float32)[None],
            jnp.asarray(global_vec, dtype=jnp.float32)[None],
        )[0]
        legal = enumerate_legal_actions(observation)
        indices = np.asarray(
            [engine_action_to_index(a.as_tuple()) for a in legal], dtype=np.int32
        )
        gathered = np.asarray(logits)[indices]
        action = legal[int(np.argmax(gathered))]
        return ActionDecision(
            action=action,
            new_state=state,
            policy_id=self.policy_id,
            strategic_option="LEARNED",
        )


def serve() -> None:
    if not PARAMS_NPZ.is_file():
        raise SystemExit(f"TEACHER-R2 BC checkpoint not found: {PARAMS_NPZ}")
    run_agent(_TeacherBcPolicy(load_params()), deterministic=True)
