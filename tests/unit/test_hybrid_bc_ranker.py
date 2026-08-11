"""Unit tests for hybrid BC ranker and proposal interface."""

from __future__ import annotations

from pathlib import Path

from generals_bot.action import KIND_MOVE, KIND_PASS, Action, PASS_ACTION
from generals_bot.models.action_index import ACTION_DIM, PASS_INDEX, action_to_index
from generals_bot.observation import GameContext, Observation
from generals_bot.policies.base import Proposal, TraceLevel
from generals_bot.policies.heuristic_v2_ablations import create_ablation
from generals_bot.policies.hybrid_bc_ranker import (
    HybridBcRankerPolicy,
    HybridConfidenceConfig,
    canonicalize_proposals,
)

BC_JSON = Path("experiments/phase9f_cnn_ranker_v1/checkpoints/bc/model.json")


def _obs() -> Observation:
    return Observation(
        height=4,
        width=4,
        turn=60,
        my_land=4,
        my_army=20,
        opp_land=2,
        opp_army=8,
        type_grid=((4, 1, 1, 0), (1, 1, 1, 1), (1, 1, 1, 1), (0, 1, 1, 1)),
        owner_grid=((1, 1, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)),
        army_grid=((8, 4, 0, 0), (3, 0, 0, 0), (2, 0, 0, 0), (0, 0, 0, 0)),
    )


def _proposal(action: Action, *, option: str, hard_priority: int, module: str = "t") -> Proposal:
    return Proposal(
        action=action,
        option=option,
        module=module,
        hard_priority=hard_priority,
        score=float(hard_priority),
        confidence=0.5,
        explanation_code="test",
    )


def test_generate_proposals_exists_on_ablation() -> None:
    policy = create_ablation("heuristic_v2f_plus_planner_terminal_fix")
    assert callable(getattr(policy, "generate_proposals", None))
    st = policy.initial_state(GameContext(0, 4, 4))
    proposals, new_state, legal = policy.generate_proposals(_obs(), st, deadline=None)
    assert isinstance(proposals, list)
    assert new_state is not None
    assert isinstance(legal, list) and legal
    d = policy.act(
        _obs(),
        policy.initial_state(GameContext(0, 4, 4)),
        deterministic=True,
        trace=TraceLevel.NONE,
        deadline=None,
    )
    assert d.action is not None


def test_canonicalize_dedupe_keeps_highest_priority_and_pass_once() -> None:
    move_a = Action(kind=KIND_MOVE, row=0, col=0, direction=1, split=0)
    props = [
        _proposal(move_a, option="EXPAND", hard_priority=10),
        _proposal(move_a, option="EXPAND", hard_priority=40),
        _proposal(PASS_ACTION, option="WAIT", hard_priority=0),
        _proposal(PASS_ACTION, option="WAIT", hard_priority=5),
        _proposal(Action(kind=KIND_PASS), option="WAIT", hard_priority=1),
    ]
    out = canonicalize_proposals(props)
    idxs = [action_to_index(p.action) for p in out]
    assert idxs.count(PASS_INDEX) == 1
    move_hits = [p for p in out if p.action.as_tuple() == move_a.as_tuple()]
    assert len(move_hits) == 1
    assert move_hits[0].hard_priority == 40
    pass_hits = [p for p in out if action_to_index(p.action) == PASS_INDEX]
    assert pass_hits[0].hard_priority == 5


def test_load_fail_falls_back_heuristic_only(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_model.json"
    policy = HybridBcRankerPolicy(checkpoint_json=missing, device="cpu")
    assert not policy.model_loaded
    st = policy.initial_state(GameContext(0, 4, 4))
    assert st.data.get("hybrid_hidden") is None
    d = policy.act(_obs(), st, deterministic=True, trace=TraceLevel.NONE, deadline=None)
    assert d.fallback_used is True
    assert (d.shield_result or {}).get("hybrid") == "heuristic_only"
    assert int(d.new_state.data.get("hybrid_forward_count") or 0) == 0


def test_recurrent_advances_on_fallback_win() -> None:
    """When model is loaded, one forward runs even if confidence/safety falls back."""
    if not BC_JSON.is_file():
        import pytest

        pytest.skip("BC checkpoint not present")

    policy = HybridBcRankerPolicy(checkpoint_json=BC_JSON, device="cpu")
    assert policy.model_loaded
    # Force confidence failure so shield fallback wins; forward must still run.
    policy.confidence = HybridConfidenceConfig(
        min_top2_margin=1.0,
        max_normalised_entropy=0.0,
        min_support_size=10_000,
    )
    st = policy.initial_state(GameContext(0, 4, 4))
    d1 = policy.act(_obs(), st, deterministic=True, trace=TraceLevel.NONE, deadline=None)
    assert int(d1.new_state.data.get("hybrid_forward_count") or 0) == 1
    assert d1.new_state.data.get("hybrid_hidden") is not None
    assert d1.fallback_used is True
    d2 = policy.act(_obs(), d1.new_state, deterministic=True, trace=TraceLevel.NONE, deadline=None)
    assert int(d2.new_state.data.get("hybrid_forward_count") or 0) == 2
    assert d2.new_state.data.get("hybrid_hidden") is not None
    assert ACTION_DIM > 0
