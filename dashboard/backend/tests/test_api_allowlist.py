"""Dashboard API allowlist and contract tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from dashboard.backend.app.main import app

client = TestClient(app)


def test_health() -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["bind"] == "127.0.0.1"


def test_job_allowlist_includes_submitted_fix_not_typos() -> None:
    res = client.get("/api/jobs/allowlist")
    assert res.status_code == 200
    body = res.json()
    assert "MATCH" in body["jobs"]
    assert "heuristic_v2f_plus_planner_terminal_fix" in body["candidates"]
    assert "heuristic_v2f_plus_planner_terminal_form" not in body["candidates"]
    assert "heuristic_v2f_plus_planner_terminal_force" not in body["candidates"]


def test_rejects_unknown_candidate() -> None:
    res = client.post(
        "/api/jobs/match",
        json={
            "job_type": "MATCH",
            "candidate": "not_a_real_bot",
            "opponent": "pass",
            "seed": 0,
            "max_turns": 5,
        },
    )
    assert res.status_code == 400


def test_rejects_path_traversal_replay() -> None:
    res = client.get("/api/replays/../secrets")
    assert res.status_code in (400, 404)


def test_overview_no_learned_champion() -> None:
    res = client.get("/api/overview")
    assert res.status_code == 200
    body = res.json()
    assert body["learned_champion"] is None
    assert body["learned_champion_note"] == "NO LEARNED CHAMPION"
    assert "elo" not in body
    assert body["active_submitted_package"]["authoritative_policy_source_commit"] == "027ff5d"
    assert body["active_submitted_package"]["embedded_metadata_status"] == "STALE"
    assert body["gate_board"]["HEURISTIC_DEVELOPMENT_GATE"] == "FAIL"
    assert body["gate_status"]["current"]["learning_readiness"] == "PASS"
    # Upload-time PENDING must not override current readiness.
    hist = body["gate_status"]["historical_observations"]
    if hist:
        assert hist[0].get("source") == "UPLOAD_RECORD"


def test_current_gates_not_overridden_by_upload_record() -> None:
    res = client.get("/api/overview")
    assert res.status_code == 200
    body = res.json()
    current = body["gate_status"]["current"]
    assert current["learning_readiness"] == "PASS"
    assert current["learned_promotion"] in {"NONE", "BLOCKED"}
    for obs in body["gate_status"]["historical_observations"]:
        # Historical may say PENDING_AT_RECORD_TIME; current must remain PASS.
        assert current["learning_readiness"] != obs.get("learning_readiness") or current[
            "learning_readiness"
        ] == "PASS"


def test_build_info_endpoint() -> None:
    res = client.get("/api/build-info")
    assert res.status_code == 200
    body = res.json()
    assert body["kind"] == "BUILD_INFO"
    assert "repository" in body
    assert "mismatch" in body


def test_capabilities_reasons() -> None:
    res = client.get("/api/capabilities")
    assert res.status_code == 200
    caps = res.json()["capabilities"]
    assert caps["portal_upload"]["enabled"] is False
    assert "manual" in caps["portal_upload"]["reason"].lower()
    assert caps["git_mutation"]["enabled"] is False
    assert caps["environment_step"]["enabled"] is True
    assert "session" in caps["environment_step"]["reason"].lower()
    assert caps["environment_reset"]["enabled"] is True


def test_competition_snapshot_not_live() -> None:
    res = client.get("/api/competition")
    assert res.status_code == 200
    body = res.json()
    snap = body.get("profile_snapshot")
    if snap:
        assert snap["live"] is False
        assert "observed_at" in snap
        assert "provenance" in snap
        assert "attribution_method" in snap
        assert "source_reference" in snap


def test_submission_metadata_authority() -> None:
    res = client.get("/api/submission")
    assert res.status_code == 200
    pkg = res.json()["package"]
    assert pkg["authoritative_policy_source_commit"] == "027ff5d"
    assert pkg["embedded_bot_commit"] == "ee06778"
    assert pkg["embedded_metadata_status"] == "STALE"
    assert res.json()["upload_enabled"] is False


def test_api_404_json() -> None:
    res = client.get("/api/this-route-does-not-exist")
    assert res.status_code == 404
    assert "detail" in res.json()


def test_spa_fallback_non_api() -> None:
    res = client.get("/overview")
    # Either index.html (200) or frontend-not-built (404) — never JSON API shape for SPA miss when dist missing
    assert res.status_code in (200, 404)
    if res.status_code == 200:
        assert "text/html" in res.headers.get("content-type", "")


def test_population_empty_state() -> None:
    res = client.get("/api/population")
    assert res.status_code == 200
    body = res.json()
    assert body["state"] in {
        "POPULATION DEVELOPMENT NOT YET RECORDED",
        "POPULATION EVIDENCE RECORDED",
    }


def test_models_graph_latency_warning() -> None:
    res = client.get("/api/models")
    assert res.status_code == 200
    assert "139" in res.json()["graph_latency_warning"]


def test_training_exposes_charts_schema() -> None:
    res = client.get("/api/training")
    assert res.status_code == 200
    body = res.json()
    assert "charts" in body
    assert "cloud_valid_learning" in body
    assert body["labels"]["charts"]
    latency = body["smoke"].get("competition_size_latency_gate")
    if latency:
        assert "classification" in latency


def test_candidate_identity_canonical_is_terminal_fix() -> None:
    res = client.get("/api/overview")
    assert res.status_code == 200
    ident = res.json()["candidate_identity"]
    assert ident["submitted_evidence_id"] == "heuristic_v2f_plus_planner_terminal_fix"
