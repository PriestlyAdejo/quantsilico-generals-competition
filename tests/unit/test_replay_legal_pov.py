"""Legal-POV reconstruction tests (ELITE_REPLAY_AUGMENTATION charter §4/§13).

Synthetic 4x4 replay; no live API. Asserts the hard fog-of-war gate:
hidden enemy state is structurally absent or stale-memory-only, never live.
"""

from __future__ import annotations

from scripts.data.replay_legal_pov import (
    PovMemory,
    legal_observation,
    memory_seen_count,
    parse_replay,
    visible_tiles,
)


def grid(rows):
    return [list(r) for r in rows]


def armies(cells: dict, size=4):
    out = [[0] * size for _ in range(size)]
    for (r, c), v in cells.items():
        out[r][c] = v
    return out


PAYLOAD = {
    "version": 1,
    "dims": [4, 4],
    "players": ["Alpha", "Beta"],
    "seed": 7,
    "mountains": [[1, 2]],
    "castles": [[2, 1]],
    "generals": [[0, 0], [3, 3]],
    "winner": 0,
    "total_ticks": 5,
    "ticks": [
        # t0: only generals owned
        {
            "owners": grid([[0, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, 1]]),
            "armies": armies({(0, 0): 1, (3, 3): 1}),
        },
        # t1: p0 -> (0,1); p1 -> (3,2)
        {
            "owners": grid([[0, 0, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, 1, 1]]),
            "armies": armies({(0, 0): 1, (0, 1): 1, (3, 2): 1, (3, 3): 1}),
        },
        # t2: p0 -> (1,1): mountain (1,2) and city (2,1) become visible
        {
            "owners": grid([[0, 0, -1, -1], [-1, 0, -1, -1], [-1, -1, -1, -1], [-1, -1, 1, 1]]),
            "armies": armies({(0, 0): 1, (0, 1): 1, (1, 1): 1, (3, 2): 1, (3, 3): 1}),
        },
        # t3: p1 captures (0,2) WITHIN p0 vision: army 5 visible
        {
            "owners": grid([[0, 0, 1, -1], [-1, 0, -1, -1], [-1, -1, -1, -1], [-1, -1, 1, 1]]),
            "armies": armies({(0, 0): 1, (0, 1): 1, (0, 2): 5, (1, 1): 1, (3, 2): 1, (3, 3): 1}),
        },
        # t4: p1 captures (0,1); (0,2) leaves p0 vision, true army now 9 (hidden)
        {
            "owners": grid([[0, 1, 1, -1], [-1, 0, -1, -1], [-1, -1, -1, -1], [-1, -1, 1, 1]]),
            "armies": armies({(0, 0): 2, (0, 1): 3, (0, 2): 9, (1, 1): 1, (3, 2): 1, (3, 3): 1}),
        },
    ],
}


def views():
    replay = parse_replay(PAYLOAD)
    memory = PovMemory()
    return replay, memory, [legal_observation(replay, memory, 0, t) for t in range(5)]


def test_parse_and_visibility_basics():
    replay = parse_replay(PAYLOAD)
    assert replay.dims == (4, 4)
    assert replay.generals == {0: (0, 0), 1: (3, 3)}
    vis = visible_tiles(PAYLOAD["ticks"][0]["owners"], 0)
    assert vis == {(0, 0), (0, 1), (1, 0)}


def test_parse_real_api_schema_variants():
    # The live API encodes dims as a dict and winner as a string.
    alt = dict(PAYLOAD)
    alt["dims"] = {"rows": 4, "cols": 4}
    alt["winner"] = "0"
    replay = parse_replay(alt)
    assert replay.dims == (4, 4)
    assert replay.winner == 0


def test_hidden_enemy_state_never_leaks():
    replay, memory, vs = views()
    # t0: enemy general and territory completely unknown
    assert 1 not in vs[0]["generals_view"]
    assert (3, 3) not in vs[0]["owner_view"]
    assert (3, 3) not in vs[0]["army_view"]
    # t3: enemy tile (0,2) is visible -> live army 5 legal
    assert vs[3]["army_view"][(0, 2)] == 5
    assert vs[3]["owner_view"][(0, 2)] == 1
    # t4: (0,2) hidden again. LIVE army is 9 - the view must NOT contain it.
    truth = PAYLOAD["ticks"][4]["armies"][0][2]
    assert truth == 9
    assert vs[4]["army_view"][(0, 2)] != truth  # stale memory only
    assert vs[4]["army_view"][(0, 2)] == 5  # last-seen value
    assert vs[4]["owner_view"][(0, 2)] == 1  # remembered ownership
    # enemy general never seen at any point
    assert all(1 not in v["generals_view"] for v in vs)
    # own armies always current (own general army grew 1 -> 2 at t4)
    assert vs[4]["army_view"][(0, 0)] == 2


def test_terrain_memory_persists_after_vision_loss():
    _, _, vs = views()
    # mountain (1,2) + city (2,1) first visible at t2
    assert (1, 2) not in vs[1]["mountains"]
    assert (1, 2) in vs[2]["mountains"]
    assert (2, 1) in vs[2]["cities"]
    # at t4 both are outside vision but remain known
    assert (1, 2) in vs[4]["mountains"]
    assert (2, 1) in vs[4]["cities"]


def test_ingestion_counter_and_idempotent_same_tick():
    replay = parse_replay(PAYLOAD)
    memory = PovMemory()
    legal_observation(replay, memory, 0, 2)
    assert memory_seen_count(memory) == 3
    legal_observation(replay, memory, 0, 2)  # same tick again: no re-ingest
    assert memory_seen_count(memory) == 3
    legal_observation(replay, memory, 0, 4)
    assert memory_seen_count(memory) == 5


def test_no_hidden_key_paths_in_view():
    """Structural guard: view fields are the whitelist, nothing else."""
    _, _, vs = views()
    allowed = {"turn", "visible", "owner_view", "army_view", "generals_view", "mountains", "cities"}
    for v in vs:
        assert set(v) == allowed
