"""Action legality audit tests (ELITE_REPLAY_AUGMENTATION BC-prep milestone)."""

from __future__ import annotations

import json

from scripts.data.replay_action_extraction import ActionEvent
from scripts.data.replay_action_legality import audit_replay, check_move
from scripts.data.replay_legal_pov import parse_replay


def grid(rows):
    return [list(r) for r in rows]


def armies(cells, size=4):
    out = [[0] * size for _ in range(4)]
    for (r, c), v in cells.items():
        out[r][c] = v
    return out


OWNERS_A = grid([[0, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, 1]])
OWNERS_B = grid([[0, 0, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, 1]])


def move_event(src, dst, amount, tick=0):
    return ActionEvent(tick, 0, "MOVE", src=src, dst=dst, amount=amount)


def test_legal_move_passes_all_rules():
    prev_armies = armies({(0, 0): 10, (3, 3): 1})
    result = check_move(move_event((0, 0), (0, 1), 6), OWNERS_A, prev_armies, set(), 0)
    assert result.legal
    assert result.violations == []


def test_src_army_one_cannot_move():
    prev_armies = armies({(0, 0): 1, (3, 3): 1})
    result = check_move(move_event((0, 0), (0, 1), 1), OWNERS_A, prev_armies, set(), 0)
    assert not result.legal
    assert "src_army_gt_1" in result.violations
    assert "amount_in_range" in result.violations


def test_amount_exceeding_army_minus_one_flagged():
    prev_armies = armies({(0, 0): 5, (3, 3): 1})
    result = check_move(move_event((0, 0), (0, 1), 5), OWNERS_A, prev_armies, set(), 0)
    assert not result.legal
    assert result.violations == ["amount_in_range"]


def test_nonadjacent_flagged():
    prev_armies = armies({(0, 0): 10, (3, 3): 1})
    result = check_move(move_event((0, 0), (0, 2), 6), OWNERS_A, prev_armies, set(), 0)
    assert not result.legal
    assert result.violations == ["adjacent"]


def test_move_into_mountain_flagged():
    prev_armies = armies({(0, 0): 10, (3, 3): 1})
    result = check_move(move_event((0, 0), (0, 1), 6), OWNERS_A, prev_armies, {(0, 1)}, 0)
    assert not result.legal
    assert result.violations == ["dst_not_mountain"]


def test_src_not_owned_flagged():
    prev_armies = armies({(0, 0): 10, (3, 3): 1})
    result = check_move(move_event((0, 0), (0, 1), 6), OWNERS_A, prev_armies, set(), 1)
    assert not result.legal
    assert "src_owned" in result.violations


def test_audit_replay_counts_legal_and_illegal_moves():
    payload = {
        "dims": {"rows": 4, "cols": 4},
        "players": ["A", "B"],
        "mountains": [],
        "castles": [],
        "generals": [[0, 0], [3, 3]],
        "winner": 0,
        "ticks": [
            {"owners": OWNERS_A, "armies": armies({(0, 0): 10, (3, 3): 1})},
            {"owners": OWNERS_B, "armies": armies({(0, 0): 4, (0, 1): 6, (3, 3): 1})},
        ],
    }
    report = audit_replay(parse_replay(payload))
    p0 = report["per_player"][0]
    assert p0["events"]["MOVE"] == 1
    assert p0["moves_legal"] == 1
    assert p0["moves_illegal"] == 0
    p1 = report["per_player"][1]
    assert p1["events"]["MOVE"] == 0  # player B only passed


def test_audit_dataset_aggregates(tmp_path):
    payload = {
        "dims": {"rows": 4, "cols": 4},
        "players": ["A", "B"],
        "mountains": [],
        "castles": [],
        "generals": [[0, 0], [3, 3]],
        "winner": 0,
        "ticks": [
            {"owners": OWNERS_A, "armies": armies({(0, 0): 10, (3, 3): 1})},
            {"owners": OWNERS_B, "armies": armies({(0, 0): 4, (0, 1): 6, (3, 3): 1})},
        ],
    }
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "1.json").write_text(json.dumps(payload), encoding="utf-8")
    from scripts.data.replay_action_legality import audit_dataset

    report = audit_dataset(tmp_path)
    assert report["totals"]["replays"] == 1
    assert report["totals"]["moves_legal"] == 1
    assert report["totals"]["moves_illegal"] == 0
