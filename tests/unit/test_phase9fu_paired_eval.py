"""Tests for resumable Phase 9FU paired evaluator (implementation amendment)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.phase9fu_paired_eval as pe


def test_parse_args_candidate_and_opponent() -> None:
    ns = pe.parse_args(["--candidate", "QS-P9FU-HYBRID-BC-V1", "--opponent", "official_hunter"])
    assert ns.candidates == ["QS-P9FU-HYBRID-BC-V1"]
    assert ns.opponents == ["official_hunter"]


def test_atomic_write_and_resume_idempotency(tmp_path: Path) -> None:
    path = tmp_path / "partial.json"
    payload = {"schema_version": 1, "completed_pair_order": ["a:1"], "completed_pairs": {"a:1": {"seed": 1}}}
    pe.atomic_write_json(path, payload)
    loaded = pe.load_checkpoint(path)
    assert loaded["completed_pair_order"] == ["a:1"]
    pe.atomic_write_json(path, payload)
    loaded2 = pe.load_checkpoint(path)
    assert loaded2 == loaded


def test_resume_protocol_mismatch_rejected(tmp_path: Path) -> None:
    partial = {
        "candidate_id": "QS-P9FU-HYBRID-BC-V1",
        "protocol_sha256": "deadbeef",
        "planned_pairs": [{"opponent": "x", "seed": 1}],
    }
    with pytest.raises(SystemExit, match="protocol hash"):
        pe.validate_resume(
            partial,
            candidate_id="QS-P9FU-HYBRID-BC-V1",
            proto_hash="cafebabe",
            planned=[("x", 1)],
        )


def test_resume_candidate_mismatch_rejected() -> None:
    partial = {
        "candidate_id": "OTHER",
        "protocol_sha256": "abc",
        "planned_pairs": [],
    }
    with pytest.raises(SystemExit, match="candidate mismatch"):
        pe.validate_resume(partial, candidate_id="QS-P9FU-HYBRID-BC-V1", proto_hash="abc", planned=[])


def test_timeout_game_not_scored_in_summary() -> None:
    pairs = [
        {
            "opponent": "official_hunter",
            "seed": 1,
            "games": [
                {"wdl": "win", "scored": True},
                {"wdl": "incomplete_timeout", "scored": False},
            ],
        }
    ]
    summary = pe._summarize(pairs)
    assert summary["wins"] == 1
    assert summary["games"] == 1
    assert summary["unscored_incomplete_timeout_games"] == 1


def test_protocol_fault_not_scored_as_pass() -> None:
    pairs = [
        {
            "opponent": "official_hunter",
            "seed": 1,
            "games": [
                {
                    "wdl": "protocol_fault",
                    "scored": False,
                    "fault_class": "PROTOCOL_FAULT",
                    "fault": {"exception_type": "RuntimeError"},
                },
                {"wdl": "win", "scored": True},
            ],
        }
    ]
    summary = pe._summarize(pairs)
    assert summary["wins"] == 1
    assert summary["games"] == 1
    assert summary["protocol_fault_games"] == 1


def test_protocol_fault_result_shape() -> None:
    out = pe._protocol_fault_result(
        seat=0,
        turn=3,
        seed=42,
        exc=RuntimeError("boom"),
        focal_seat=0,
        turns=3,
    )
    assert out["fault_class"] == "PROTOCOL_FAULT"
    assert out["scored"] is False
    assert out["fault"]["exception_type"] == "RuntimeError"
    assert len(out["fault"]["traceback_sha256"]) == 64


def test_classify_incomplete_timeout() -> None:
    thr = {
        "min_direct_score_rate_improvement_vs_v001": 0.05,
        "max_opponent_suite_score_rate_regression": 0.05,
        "max_draw_rate_increase": 0.15,
    }
    cand = {
        "status": "INCOMPLETE_TIMEOUT",
        "suite_score_rate_mean": 0.5,
        "direct_vs_v001": {
            "summary": {"pairs": 2, "score_rate": 0.6, "draw_rate": 0.1, "wins": 1, "draws": 1, "losses": 0}
        },
    }
    out = pe.classify_candidate(cand, thr, n_direct=16)
    assert out["label"] == "ABORTED_INCOMPLETE"


def test_planned_pairs_respects_opponent_filter() -> None:
    proto = json.loads(pe.PROTOCOL.read_text(encoding="utf-8"))
    planned = pe.planned_pairs(proto, opponents=["official_hunter"])
    assert planned
    assert all(o == "official_hunter" for o, _ in planned)


def test_aggregate_from_partial_skips_missing(tmp_path: Path) -> None:
    proto = json.loads(pe.PROTOCOL.read_text(encoding="utf-8"))
    partial = {
        "status": "COMPLETE",
        "completed_pairs": {
            "heuristic_v2f_plus_planner_terminal_fix:101": {
                "opponent": "heuristic_v2f_plus_planner_terminal_fix",
                "seed": 101,
                "games": [
                    {"wdl": "win", "scored": True},
                    {"wdl": "loss", "scored": True},
                ],
            }
        },
        "completed_pair_order": ["heuristic_v2f_plus_planner_terminal_fix:101"],
    }
    agg = pe.aggregate_from_partial(partial, proto)
    assert agg["direct_vs_v001"]["summary"]["games"] == 2
    assert agg["completed_pairs_n"] == 1


def test_heartbeat_emits(monkeypatch: pytest.MonkeyPatch) -> None:
    hb = pe.Heartbeat(interval_s=0.05)
    hb.update(candidate="X", phase="test")
    hb.start()
    import time

    time.sleep(0.18)
    hb.stop()
    assert hb.emissions >= 1
