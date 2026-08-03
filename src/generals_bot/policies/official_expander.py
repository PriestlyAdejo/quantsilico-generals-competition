"""Official Expander competition agent wrapped as a Policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from generals_bot.action import Action
from generals_bot.observation import GameContext, Observation
from generals_bot.policies.base import ActionDecision, PolicyState, TraceLevel

_ENGINE_EXPANDER = (
    Path(__file__).resolve().parents[3]
    / "third_party"
    / "generals-bots"
    / "competition"
    / "agents"
    / "expander_python"
    / "agent.py"
)


def _load_expander_cls():
    spec = importlib.util.spec_from_file_location("official_expander_agent", _ENGINE_EXPANDER)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load expander from {_ENGINE_EXPANDER}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Agent


class OfficialExpanderPolicy:
    """Adapter around competition/agents/expander_python/agent.py."""

    policy_id = "official_expander"

    def __init__(self) -> None:
        self._Agent = _load_expander_cls()
        self._agent = None

    def initial_state(self, context: GameContext) -> PolicyState:
        self._agent = self._Agent(context.player_id, context.height, context.width)
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
        assert self._agent is not None
        obs = SimpleNamespace(
            H=observation.height,
            W=observation.width,
            turn=observation.turn,
            my_land=observation.my_land,
            my_army=observation.my_army,
            opp_land=observation.opp_land,
            opp_army=observation.opp_army,
            type_grid=observation.type_grid,
            owner_grid=observation.owner_grid,
            army_grid=observation.army_grid,
        )
        kind, row, col, direction, split = self._agent.act(obs)
        action = Action(kind=int(kind), row=int(row), col=int(col), direction=int(direction), split=int(split))
        return ActionDecision(
            action=action,
            new_state=state,
            strategic_option="EXPAND",
            policy_id=self.policy_id,
            legal_action_count=-1,
        )
