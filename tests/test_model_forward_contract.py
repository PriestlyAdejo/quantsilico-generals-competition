"""Tests for typed model-forward contract and protocol fault denominator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from generals_bot.models.factory import build_model
from generals_bot.models.model_forward import (
    MalformedModelOutputError,
    adapt_forward_output,
)

REPO = Path(__file__).resolve().parents[1]


def test_adapt_forward_rejects_tuple_indexing_path() -> None:
    with pytest.raises(MalformedModelOutputError, match="sequence"):
        adapt_forward_output((torch.zeros(1, 3), torch.zeros(1, 1), torch.zeros(1, 4)))


def test_adapt_forward_requires_dict_keys() -> None:
    with pytest.raises(MalformedModelOutputError, match="missing"):
        adapt_forward_output({"hidden": torch.zeros(1, 4)})


def test_cnn_and_graph_forward_through_adapter() -> None:
    from generals_bot.models.observation_encoder import GLOBAL_DIM, MAX_HW, NUM_CELL_CHANNELS

    for arch in ("recurrent_cnn_v2", "recurrent_graph_belief_v2"):
        model = build_model(arch).eval()
        cells = torch.zeros(1, NUM_CELL_CHANNELS, MAX_HW, MAX_HW)
        glob = torch.zeros(1, GLOBAL_DIM)
        hidden = model.initial_hidden(1)
        kwargs = {"deterministic": True}
        if hasattr(model, "initial_cell_memory"):
            raw = model.forward_tensors(cells, glob, hidden, model.initial_cell_memory(1), **kwargs)
        else:
            raw = model.forward_tensors(cells, glob, hidden, **kwargs)
        fwd = adapt_forward_output(raw)
        assert fwd.logits.ndim >= 1
        assert fwd.hidden.shape[0] == 1
        assert fwd.value.shape[0] == 1


def test_initial_fault_denominator_matches_two_full_games() -> None:
    path = REPO / "experiments" / "manifests" / "adaptive_initial_campaign.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for cand in data["candidates"]:
        hist = cand["validation_history"]
        assert len(hist) >= 1
        # Each recorded eval: 2 games × 100 turns → 200 faults under per-turn exception hypothesis
        for row in hist:
            assert row["games"] == 2
            assert row["protocol_faults"] == 200
            assert row["wins"] == 0 and row["draws"] == 2 and row["losses"] == 0
            assert row["score_rate"] == 0.5
        # Denominator: faults / (games * max_turns) == 1.0 on the learned side
        max_turns = 100
        faults_per_decision = 200 / (2 * max_turns)
        assert faults_per_decision == 1.0
