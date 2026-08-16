"""Exact action derivation from elite replays via the pinned engine oracle.

EV-0042 milestone: replace heuristic candidate actions with ENGINE-VERIFIED
derivations. For each tick t, reconstruct TRUE_COMPETITION_STATE, generate a
bounded candidate action set per player around the heuristic extraction
(plus pass), and search for the action PAIR whose pinned-engine step
reproduces the recorded tick t+1 (armies + ownership within the real board).

Authority (charter §0): the pinned engine is the oracle. Matches are exact;
non-matches are reported with categories, never papered over. Runner faults
and timing glitches are NOT reconstructable from replay payloads - the
unmatched category records them as possibilities, not verdicts.

The search never consults hidden information about INTENT: candidates are
built from full state (oracle-legal) but classification of matched actions
vs the player's legal observation is the SEPARATE replay_legal_pov pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from generals.core import game

from scripts.data.replay_action_extraction import extract_tick_actions
from scripts.data.replay_engine_oracle import (
    ENGINE_SUBMODULE_SHA,
    RULESET,
    state_from_tick,
)
from scripts.data.replay_legal_pov import parse_replay

PASS = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
MAX_CANDIDATES_PER_PLAYER = 11  # pass + heuristic + 2 splits x 4 dirs + 2 dst-follow


@jax.jit
def _step_pairs(state, pairs):
    """State passed as an ARGUMENT so jit caches one executable per batch size."""
    return jax.vmap(lambda a: game.step(state, a)[0])(pairs)


def _heuristic_move(tick, nxt, replay, player: int, t: int) -> tuple | None:
    events = extract_tick_actions(
        tick["owners"], tick["armies"], nxt["owners"], nxt["armies"],
        replay.cities, player, t,
    )
    for ev in events:
        if ev.kind == "MOVE" and ev.src is not None and ev.dst is not None:
            return ev.src
    return None


def _dst_follow(tick, nxt, replay, player: int) -> tuple | None:
    """Largest ownership/army gain on a non-city tile (heuristic v1 dst)."""
    best, best_gain = None, 0
    cities = replay.cities
    for r, row in enumerate(nxt["owners"]):
        for c, owner in enumerate(row):
            pos = (r, c)
            if pos in cities or tick["owners"][r][c] == player:
                continue
            base = tick["armies"][r][c] if tick["owners"][r][c] == player else 0
            gain = nxt["armies"][r][c] - base
            if owner == player and gain > best_gain:
                best, best_gain = pos, gain
    return best


def candidate_actions(tick, nxt, replay, player: int, t: int) -> list[jnp.ndarray]:
    out = [PASS]
    src = _heuristic_move(tick, nxt, replay, player, t)
    dst = _dst_follow(tick, nxt, replay, player)
    anchors = [p for p in (src, dst) if p is not None]
    seen: set[tuple] = set()
    for anchor in anchors:
        for direction in range(4):
            for split in (0, 1):
                action = (0, anchor[0], anchor[1], direction, split)
                if action not in seen:
                    seen.add(action)
                    out.append(jnp.asarray(action, dtype=jnp.int32))
        if len(out) >= MAX_CANDIDATES_PER_PLAYER:
            break
    return out[:MAX_CANDIDATES_PER_PLAYER]


def _parity_batch(
    new_armies, new_ownership, nxt: dict, h: int, w: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorised parity over N stepped pair-candidates."""
    pred_armies = new_armies[:, :h, :w]
    true_armies = jnp.asarray(nxt["armies"], dtype=jnp.int32)[None]
    armies_ok = jnp.all(pred_armies == true_armies, axis=(1, 2))
    owners_ok = jnp.ones(new_armies.shape[0], dtype=bool)
    for player in (0, 1):
        pred = new_ownership[:, player, :h, :w]
        true_own = jnp.asarray(
            [[o == player for o in row] for row in nxt["owners"]]
        )[None]
        owners_ok = owners_ok & jnp.all(pred == true_own, axis=(1, 2))
    return armies_ok, owners_ok


def derive_tick(
    replay, t: int, time_phase: int = 0
) -> dict:
    tick, nxt = replay.ticks[t], replay.ticks[t + 1]
    true_state = state_from_tick(
        tick,
        dims=replay.dims,
        mountains=replay.mountains,
        castles=replay.cities,
        generals=replay.generals,
        time=t + time_phase,
    )
    eng = true_state.engine_state
    h, w = replay.dims
    cands = [
        candidate_actions(tick, nxt, replay, player, t) for player in range(2)
    ]
    pairs = jnp.stack(
        [jnp.stack([a0, a1]) for a0 in cands[0] for a1 in cands[1]]
    )  # (N, 2, 5)
    new_states = _step_pairs(eng, pairs)
    armies_ok, owners_ok = _parity_batch(new_states.armies, new_states.ownership, nxt, h, w)
    exact_idx = jnp.argmax(armies_ok & owners_ok)
    if bool((armies_ok & owners_ok)[int(exact_idx)]):
        status = "EXACT_MATCH"
    else:
        owners_idx = jnp.argmax(owners_ok)
        if bool(owners_ok[int(owners_idx)]):
            status, exact_idx = "OWNERS_ONLY", owners_idx
        else:
            return {"status": "NO_MATCH", "actions": None, "tick": t}
    pair = pairs[int(exact_idx)]
    return {
        "status": status,
        "actions": [tuple(int(x) for x in pair[0]), tuple(int(x) for x in pair[1])],
        "tick": t,
    }


def derive_replay(payload: dict) -> dict:
    replay = parse_replay(payload)
    counts = {"EXACT_MATCH": 0, "OWNERS_ONLY": 0, "NO_MATCH": 0}
    for t in range(len(replay.ticks) - 1):
        result = derive_tick(replay, t)
        counts[result["status"]] += 1
    return {"ticks": len(replay.ticks) - 1, "counts": counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit-replays", type=int, default=None)
    args = parser.parse_args()
    raw_dir = Path(args.dataset_dir) / "raw"
    report = {
        "engine_sha": ENGINE_SUBMODULE_SHA,
        "ruleset": RULESET,
        "derivation_version": "exact-derivation/1.0",
        "dataset_id": Path(args.dataset_dir).name,
        "replays": {},
        "aggregate": {"EXACT_MATCH": 0, "OWNERS_ONLY": 0, "NO_MATCH": 0, "ticks": 0},
    }
    paths = sorted(raw_dir.glob("*.json"))
    if args.limit_replays:
        paths = paths[: args.limit_replays]
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = derive_replay(payload)
        report["replays"][path.name] = result
        for key, value in result["counts"].items():
            report["aggregate"][key] += value
        report["aggregate"]["ticks"] += result["ticks"]
        print(f"{path.name}: {result['counts']}")
    agg = report["aggregate"]
    print(
        f"aggregate: exact={agg['EXACT_MATCH']}/{agg['ticks']} "
        f"owners_only={agg['OWNERS_ONLY']} no_match={agg['NO_MATCH']}"
    )
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
