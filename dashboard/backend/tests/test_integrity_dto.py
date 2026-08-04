"""Dashboard integrity DTO tests — qualification, docs, repository, env sessions."""

from __future__ import annotations

from fastapi.testclient import TestClient

from dashboard.backend.app.main import app

client = TestClient(app)


def test_qualification_development_metrics() -> None:
    res = client.get("/api/qualification")
    assert res.status_code == 200
    body = res.json()
    assert body["default_candidate"] == "heuristic_v2f_plus_planner_terminal_fix"
    assert "terminal_form" not in body["default_candidate"]
    cand = body["candidates"][0]
    assert cand["id"] == "heuristic_v2f_plus_planner_terminal_fix"
    dev = cand["development_wdl"]
    assert dev["availability"] == "RECORDED"
    assert (dev["wins"], dev["draws"], dev["losses"]) == (21, 27, 0)
    assert cand["discovery"]["availability"] == "RECORDED"
    assert abs(float(cand["discovery"]["value"]) - 0.4375) < 1e-9
    assert cand["conversion"]["availability"] == "RECORDED"
    assert float(cand["conversion"]["value"]) == 1.0
    screening = cand.get("screening_wdl") or {}
    assert screening.get("availability") == "MISSING"
    assert any(s.get("label") == "Development Evaluation" for s in body["stages"])

def test_documentation_index_and_section() -> None:
    res = client.get("/api/documentation")
    assert res.status_code == 200
    sections = res.json()["sections"]
    assert len(sections) >= 24
    ids = {s["id"] for s in sections}
    assert "glossary" in ids
    assert "startup" in ids
    assert "arena" in ids
    sec = client.get("/api/documentation/startup")
    assert sec.status_code == 200
    body = sec.json()
    assert "body text is not served yet" not in body["content"].lower()
    assert len(body["content"]) > 40
    missing = client.get("/api/documentation/../secrets")
    assert missing.status_code == 404


def test_repository_no_skipped_generic() -> None:
    res = client.get("/api/repository")
    assert res.status_code == 200
    body = res.json()
    blob = str(body).lower()
    # Never use skipped as a generic fallback status field value
    for key in ("linux_parity", "dashboard_tests", "training_tests", "windows_validation", "ci_runs_status"):
        val = str(body.get(key, "")).upper()
        assert val != "SKIPPED"
        assert val in {
            "PASS",
            "FAIL",
            "NOT_RUN",
            "NOT_RECORDED",
            "NOT_CONFIGURED",
            "NOT_APPLICABLE",
            "UNKNOWN",
            "",
        } or key == "windows_validation"
    assert body.get("branch")
    assert body.get("commit")
    assert body.get("engine_commit")
    assert isinstance(body.get("recent_commits"), list)
    assert body.get("mutations", {}).get("enabled") is False
    assert "skipped" not in blob or "not skipped" in blob  # allow prose denying skipped


def test_environment_session_create_and_close() -> None:
    caps = client.get("/api/environment")
    assert caps.status_code == 200
    assert caps.json()["capabilities"]["sessions"] is True
    created = client.post("/api/environment/sessions", json={"seed": 7, "map_preset": "standard", "ttl_s": 900})
    # Session create may fail if GeneralsEnv import/runtime unavailable in this env;
    # treat hard 500 as recorded failure for the gate runner rather than silent skip.
    if created.status_code == 429:
        return
    assert created.status_code == 200, created.text
    sid = created.json()["session_id"]
    got = client.get(f"/api/environment/sessions/{sid}")
    assert got.status_code == 200
    assert got.json()["board"]["width"] >= 1
    closed = client.delete(f"/api/environment/sessions/{sid}")
    assert closed.status_code == 200
