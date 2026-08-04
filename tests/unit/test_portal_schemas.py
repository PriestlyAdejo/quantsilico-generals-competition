"""Portal attribution schema smoke tests."""

from generals_bot.schemas import MatchResultRecord
from generals_bot.schemas.portal import (
    GateStatusBoard,
    PortalMatchObservation,
    PortalSubmissionVersion,
)


def test_portal_match_inferred_attribution_warns() -> None:
    obs = PortalMatchObservation(
        portal_match_id="m1",
        attribution_method="INFERRED_ACTIVE_UPLOAD_WINDOW",
        attribution_confidence="LOW",
    )
    d = obs.to_dict()
    assert "attribution_warning" in d
    assert "not exact" in d["attribution_warning"]


def test_portal_submission_exact_methods_do_not_warn() -> None:
    ver = PortalSubmissionVersion(
        portal_submission_label="heuristic_v2_preppo",
        package_sha256="abc",
        attribution_method="MANUAL_OPERATOR_ASSIGNMENT",
        attribution_confidence="HIGH",
    )
    assert ver.attribution_warning() is None


def test_match_result_record_has_portal_fields() -> None:
    rec = MatchResultRecord(
        portal_match_id="123",
        package_sha256="e123",
        attribution_method="UNATTRIBUTED",
    )
    d = rec.to_dict()
    assert d["portal_match_id"] == "123"
    assert d["package_sha256"] == "e123"


def test_gate_status_board_names() -> None:
    board = GateStatusBoard(
        HEURISTIC_DEVELOPMENT_GATE="FAIL",
        PRE_PPO_SUBMISSION_GATE="PASS",
        PORTAL_SUBMISSION_GATE="PASS",
        LEARNING_READINESS_GATE="NOT_RUN",
        LEARNED_PROMOTION_GATE="NONE",
        final_tournament_qualified=False,
    )
    d = board.to_dict()
    assert d["PORTAL_SUBMISSION_GATE"] == "PASS"
    assert d["final_tournament_qualified"] is False
