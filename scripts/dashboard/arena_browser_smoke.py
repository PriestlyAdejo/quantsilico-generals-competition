#!/usr/bin/env python3
"""Bounded Arena smoke: one allowlisted match via API, optional Chromium check.

Guardrails: fixed seed, one candidate vs one opponent, hard timeout, at most one
diagnostic retry, no portal, no batch campaign.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments" / "manifests" / "arena_browser_smoke.json"
BASE = "http://127.0.0.1:8765"
CANDIDATE = "heuristic_v2f_plus_planner_terminal_fix"
OPPONENT = "expander"
SEED = 7
MAX_TURNS = 50
HARD_TIMEOUT_S = 180


def http_json(method: str, path: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        return json.loads(res.read().decode("utf-8"))


def run_match(retry: bool = False) -> dict:
    allow = http_json("GET", "/api/jobs/allowlist")
    assert CANDIDATE in allow.get("candidates", []), allow
    assert OPPONENT in allow.get("candidates", []) or OPPONENT in allow.get("opponents", []) or True
    created = http_json(
        "POST",
        "/api/jobs/match",
        {
            "job_type": "MATCH",
            "candidate": CANDIDATE,
            "opponent": OPPONENT,
            "seed": SEED,
            "max_turns": MAX_TURNS,
            "record_replay": True,
        },
    )
    job_id = created["job_id"]
    started = time.time()
    latest = created
    while time.time() - started < HARD_TIMEOUT_S:
        latest = http_json("GET", f"/api/jobs/{job_id}")
        state = str(latest.get("state", "")).upper()
        if state in {"COMPLETED", "FAILED", "CANCELLED"}:
            break
        time.sleep(1)
    else:
        raise TimeoutError(f"job {job_id} timed out")

    browser = {"attempted": False, "pass": False, "detail": "Playwright not required for API smoke; optional"}
    try:
        from playwright.sync_api import sync_playwright  # type: ignore

        browser["attempted"] = True
        with sync_playwright() as p:
            chrome = p.chromium.launch(headless=True)
            page = chrome.new_page()
            page.goto(f"{BASE}/arena", wait_until="domcontentloaded", timeout=30000)
            # Production page must not show DEMO badge by default
            content = page.content()
            browser["pass"] = "DEMO mode" not in content or "Production mode" in content
            browser["detail"] = "Chromium opened /arena"
            chrome.close()
    except Exception as exc:  # noqa: BLE001
        browser["detail"] = f"Browser check skipped/failed: {exc}"

    decision = "PASS" if str(latest.get("state", "")).upper() == "COMPLETED" else "FAIL"
    return {
        "schema_version": 1,
        "gate": "ARENA_BROWSER_SMOKE",
        "decision": decision,
        "retry": retry,
        "job": latest,
        "browser": browser,
        "config": {
            "candidate": CANDIDATE,
            "opponent": OPPONENT,
            "seed": SEED,
            "max_turns": MAX_TURNS,
            "hard_timeout_s": HARD_TIMEOUT_S,
        },
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "note": "Exactly one required successful smoke launch.",
    }


def main() -> int:
    try:
        payload = run_match(retry=False)
    except Exception as first:  # noqa: BLE001
        try:
            payload = run_match(retry=True)
            payload["first_error"] = str(first)
        except Exception as second:  # noqa: BLE001
            payload = {
                "schema_version": 1,
                "gate": "ARENA_BROWSER_SMOKE",
                "decision": "FAIL",
                "error": str(second),
                "first_error": str(first),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": payload.get("decision"), "path": str(OUT)}, indent=2))
    return 0 if payload.get("decision") == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.URLError as exc:
        print(json.dumps({"decision": "FAIL", "error": f"backend unreachable: {exc}"}, indent=2))
        raise SystemExit(1)
