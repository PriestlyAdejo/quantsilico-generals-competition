"""Dashboard API allowlist tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from dashboard.backend.app.main import app

client = TestClient(app)


def test_health() -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["bind"] == "127.0.0.1"


def test_job_allowlist() -> None:
    res = client.get("/api/jobs/allowlist")
    assert res.status_code == 200
    body = res.json()
    assert "MATCH" in body["jobs"]
    assert "heuristic_v1" in body["candidates"]


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


def test_overview_champion() -> None:
    res = client.get("/api/overview")
    assert res.status_code == 200
    assert res.json()["champion"] == "heuristic_v1"
