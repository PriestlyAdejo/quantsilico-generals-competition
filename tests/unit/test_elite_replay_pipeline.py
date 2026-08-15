"""Bounded tests for the elite-replay pilot pipeline (charter §13).

No live API calls: get_json/fetch_board are monkeypatched with fixtures.
Covers leaderboard parsing, seat-order dedup, per-player caps, dataset
version allocation, resume behaviour, and manifest content hashes.
"""

from __future__ import annotations

import json
from datetime import UTC

import pytest

from scripts.data import capture_leaderboard_snapshot as snap_mod
from scripts.data import fetch_elite_replays as fetch_mod

BOARD = {
    "leaderboard": [
        {"kind": "user", "label": "Alpha", "rank": 1},
        {"kind": "user", "label": "Beta", "rank": 2},
        {"kind": "user", "label": "Gamma", "rank": 3},
    ]
}

MATCHES_ALPHA = [
    {"id": "10", "a_name": "Alpha", "b_name": "Beta", "seed": 1, "turns": 100},
    # Seat-order duplicate of the game above: same seed + same pair.
    {"id": "11", "a_name": "Beta", "b_name": "Alpha", "seed": 1, "turns": 100},
    {"id": "12", "a_name": "Alpha", "b_name": "Gamma", "seed": 2, "turns": 80},
    {"id": "13", "a_name": "Alpha", "b_name": "Beta", "seed": 3, "turns": 5},  # too short
]
MATCHES_BETA = [
    {"id": "20", "a_name": "Beta", "b_name": "Gamma", "seed": 4, "turns": 60},
    {"id": "21", "a_name": "Beta", "b_name": "Gamma", "seed": 5, "turns": 70},
]


def fake_get_json(url: str, retries: int = 4):
    if "replay=" in url:
        rid = url.rsplit("=", 1)[1]
        return {"id": rid, "ticks": list(range(120)), "meta": {"pad": "x" * 200}}
    if "player=Alpha" in url:
        return {"matches": MATCHES_ALPHA}
    if "player=Beta" in url:
        return {"matches": MATCHES_BETA}
    if "player=Gamma" in url:
        return {"matches": []}
    return BOARD


def test_select_matches_dedups_seat_order_and_caps(monkeypatch):
    monkeypatch.setattr(fetch_mod, "get_json", fake_get_json)
    matches, counts = fetch_mod.select_matches(["Alpha", "Beta"], 2, min_turns=40)
    ids = [m["id"] for m in matches]
    assert ids == ["10", "12", "20", "21"]  # '11' deduped, '13' below min turns
    assert counts["Alpha"] == 2
    assert counts["Beta"] == 2


def test_fetch_end_to_end_manifest_and_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_mod, "get_json", fake_get_json)
    monkeypatch.setattr(
        "sys.argv",
        [
            "fetch_elite_replays.py",
            "--top",
            "3",
            "--per-player",
            "2",
            "--min-turns",
            "40",
            "--workers",
            "1",
            "--out",
            str(tmp_path),
        ],
    )
    assert fetch_mod.main() == 0
    datasets = sorted(p.name for p in tmp_path.iterdir())
    assert len(datasets) == 1
    assert datasets[0].startswith("DATASET-ELITE-")
    assert datasets[0].endswith("-V01")
    first = tmp_path / datasets[0]
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_id"].endswith("-V01")
    assert len(manifest["payload_hashes"]) == 4
    assert manifest["filtering"]["dedup_policy"].startswith("seed+sorted_player_pair")
    # Content hash is a function of the payload hashes.
    import hashlib

    expected = hashlib.sha256(
        json.dumps(manifest["payload_hashes"], sort_keys=True).encode()
    ).hexdigest()
    assert manifest["content_hash"] == expected
    # Resume: second run allocates V02 (snapshot never overwritten).
    assert fetch_mod.main() == 0
    datasets = sorted(p.name for p in tmp_path.iterdir())
    assert len(datasets) == 2
    assert datasets[1].endswith("-V02")


def test_corrupt_or_empty_payload_not_counted_present(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_mod, "get_json", fake_get_json)
    from datetime import datetime

    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    raw = tmp_path / f"DATASET-ELITE-{date_str}-V01" / "raw"
    raw.mkdir(parents=True)
    (raw / "10.json").write_text("tiny")  # <=100 bytes: treated as absent
    monkeypatch.setattr(
        "sys.argv",
        [
            "fetch_elite_replays.py",
            "--players",
            "Alpha",
            "--per-player",
            "2",
            "--min-turns",
            "40",
            "--workers",
            "1",
            "--out",
            str(tmp_path),
        ],
    )
    assert fetch_mod.main() == 0
    payload = raw / "10.json"
    assert payload.stat().st_size > 100  # refetched atomically


def test_leaderboard_snapshot_records_only_exposed_scalars(tmp_path, monkeypatch):
    board = {
        "leaderboard": [
            {"kind": "user", "label": "Alpha", "nested": {"hidden": 1}, "rating": 1500},
        ]
    }
    monkeypatch.setattr(snap_mod, "fetch_board", lambda: board)
    out = tmp_path / "snapshots.jsonl"
    monkeypatch.setattr(
        "sys.argv", ["capture_leaderboard_snapshot.py", "--out", str(out)]
    )
    assert snap_mod.main() == 0
    record = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    entry = record["entries"][0]
    assert entry["label"] == "Alpha"
    assert entry["rating"] == 1500
    assert "nested" not in entry  # non-scalar fields never persisted
    assert entry["rank_observed"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
