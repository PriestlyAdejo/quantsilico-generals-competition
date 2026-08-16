"""Action extraction tests (ELITE_REPLAY_AUGMENTATION data-plane milestone)."""

from __future__ import annotations

from scripts.data.replay_action_extraction import (
    extract_player_actions,
    extract_tick_actions,
)
from scripts.data.replay_legal_pov import parse_replay


def grid(rows):
    return [list(r) for r in rows]


def armies(cells, size=4):
    out = [[0] * size for _ in range(size)]
    for (r, c), v in cells.items():
        out[r][c] = v
    return out


OWNERS_A = grid([[0, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, 1]])
OWNERS_B = grid([[0, 0, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, 1]])


def test_adjacent_move_detected_with_amount():
    prev_armies = armies({(0, 0): 10, (3, 3): 1})
    next_armies = armies({(0, 0): 4, (0, 1): 6, (3, 3): 1})
    events = extract_tick_actions(OWNERS_A, prev_armies, OWNERS_B, next_armies, set(), 0, tick=0)
    assert len(events) == 1
    ev = events[0]
    assert (ev.kind, ev.src, ev.dst) == ("MOVE", (0, 0), (0, 1))
    assert ev.amount == 6
    assert ev.legal_pov


def test_pass_when_nothing_changes():
    a = armies({(0, 0): 5, (3, 3): 1})
    events = extract_tick_actions(OWNERS_A, a, OWNERS_A, a, set(), 0, tick=0)
    assert [e.kind for e in events] == ["PASS"]


def test_city_capture_is_build():
    owners_c = grid([[0, -1, -1, -1], [-1, 0, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, 1]])
    prev_armies = armies({(0, 0): 8, (3, 3): 1})
    next_armies = armies({(0, 0): 8, (1, 1): 3, (3, 3): 1})
    events = extract_tick_actions(OWNERS_A, prev_armies, owners_c, next_armies, {(1, 1)}, 0, tick=0)
    assert "BUILD" in [e.kind for e in events]


def test_nonadjacent_flagged_not_legal_pov():
    # src (0,0) -> dst (0,2) skipping (0,1): heuristic keeps flag False
    owners_c = grid([[0, -1, 0, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, 1]])
    prev_armies = armies({(0, 0): 10, (3, 3): 1})
    next_armies = armies({(0, 0): 3, (0, 2): 7, (3, 3): 1})
    events = extract_tick_actions(OWNERS_A, prev_armies, owners_c, next_armies, set(), 0, tick=0)
    moves = [e for e in events if e.kind == "MOVE"]
    assert len(moves) == 1
    assert moves[0].legal_pov is False


def test_full_replay_timeline_length_and_players():
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
    replay = parse_replay(payload)
    p0 = extract_player_actions(replay, 0)
    p1 = extract_player_actions(replay, 1)
    assert any(e.kind == "MOVE" and e.dst == (0, 1) for e in p0)
    assert all(e.kind == "PASS" for e in p1)  # player 1 did nothing
