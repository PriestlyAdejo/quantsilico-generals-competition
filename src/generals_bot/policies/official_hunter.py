"""Official Hunter agent wrapped as a Policy (informational local opponent)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from generals.agents import HunterAgent
from generals.core.observation import Observation as JaxObservation

from generals_bot.action import Action
from generals_bot.observation import GameContext, Observation
from generals_bot.policies.base import ActionDecision, PolicyState, TraceLevel
from generals_bot.protocol import (
    OWNER_ME,
    OWNER_NEUTRAL,
    OWNER_OPP,
    TYPE_CASTLE,
    TYPE_FOG,
    TYPE_GENERAL,
    TYPE_MOUNTAIN,
    TYPE_STRUCTURE_IN_FOG,
)


def _to_jax_observation(obs: Observation) -> JaxObservation:
    h, w = obs.height, obs.width
    armies = np.asarray(obs.army_grid, dtype=np.int32)
    types = np.asarray(obs.type_grid, dtype=np.int32)
    owners = np.asarray(obs.owner_grid, dtype=np.int32)
    fog = types == TYPE_FOG
    mountains = types == TYPE_MOUNTAIN
    castles = types == TYPE_CASTLE
    generals = types == TYPE_GENERAL
    structures_fog = types == TYPE_STRUCTURE_IN_FOG
    owned = owners == OWNER_ME
    opp = owners == OWNER_OPP
    neutral = (owners == OWNER_NEUTRAL) & ~fog & ~mountains & ~structures_fog
    return JaxObservation(
        armies=jnp.asarray(armies),
        generals=jnp.asarray(generals),
        castles=jnp.asarray(castles),
        mountains=jnp.asarray(mountains),
        neutral_cells=jnp.asarray(neutral),
        owned_cells=jnp.asarray(owned),
        opponent_cells=jnp.asarray(opp),
        fog_cells=jnp.asarray(fog),
        structures_in_fog=jnp.asarray(structures_fog),
        owned_land_count=jnp.int32(obs.my_land),
        owned_army_count=jnp.int32(obs.my_army),
        opponent_land_count=jnp.int32(obs.opp_land),
        opponent_army_count=jnp.int32(obs.opp_army),
        timestep=jnp.int32(obs.turn),
    )


class OfficialHunterPolicy:
    """Adapter around generals.agents.HunterAgent for local informational suites."""

    policy_id = "official_hunter"

    def __init__(self) -> None:
        self._agent = HunterAgent(id="Hunter")
        self._key = jnp.array([0, 0], dtype=jnp.uint32)

    def initial_state(self, context: GameContext) -> PolicyState:
        self._agent.reset()
        return PolicyState(data={"player_id": context.player_id})

    def act(
        self,
        observation: Observation,
        state: PolicyState,
        *,
        deterministic: bool,
        trace: TraceLevel,
        deadline: float | None,
    ) -> ActionDecision:
        jax_obs = _to_jax_observation(observation)
        raw = np.asarray(self._agent.act(jax_obs, self._key), dtype=np.int32)
        # Hunter returns [pass_flag, row, col, direction, split] where pass_flag True means wait.
        if int(raw[0]) == 1:
            from generals_bot.action import PASS_ACTION

            action = PASS_ACTION
        else:
            action = Action(
                kind=0,
                row=int(raw[1]),
                col=int(raw[2]),
                direction=int(raw[3]),
                split=int(raw[4]),
            )
        return ActionDecision(
            action=action,
            new_state=state,
            strategic_option="HUNT",
            policy_id=self.policy_id,
            legal_action_count=-1,
        )
