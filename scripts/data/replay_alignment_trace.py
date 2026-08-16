"""Replay/engine timing-alignment trace (competition authority, EV-0042).

Operator amendment §11: BEFORE auditing thousands of actions, prove when each
replay tick is recorded relative to action submission/resolution/growth by
stepping the PINNED competition engine from reconstructed tick T with the
extracted candidate actions and comparing against the replay's TRUE tick
T+1. No heuristic may paper over a mismatch: this probe reports raw parity.

Assumption under test: one replay tick == one engine step (both players act
per tick) with growth applied by the engine at the new time. The probe
sweeps small time-phase offsets because the recording clock may lead/lag
the engine growth phase, and reports which phase maximises exact parity.

Full-state reconstruction is oracle-only (never policy input); candidate
actions come from the existing heuristic extractor (MOVE events only; PASS
otherwise), so residual mismatch categories include extraction error, split
ambiguity, and simultaneous-combat order effects - all reported, none hidden.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
from generals.core import game

from scripts.data.replay_action_extraction import extract_tick_actions
from scripts.data.replay_engine_oracle import (
    ENGINE_SUBMODULE_SHA,
    RULESET,
    direction_from,
    state_from_tick,
)
from scripts.data.replay_legal_pov import parse_replay

PASS = (1, 0, 0, 0, 0)


def candidate_action(events, h: int, w: int) -> tuple[int, int, int, int, int]:
    """Heuristic candidate action for one player at one tick (audit trace).

    MOVE with adjacent endpoints -> protocol move (split 0); everything else
    -> explicit pass. Split ambiguity is a known residual category.
    """
    for ev in events:
        if ev.kind == "MOVE" and ev.src is not None and ev.dst is not None:
            d = direction_from(ev.src, ev.dst)
            if d is not None:
                return (0, ev.src[0], ev.src[1], d, 0)
    return PASS


def parity_at(replay, t: int, time_phase: int):
    tick, nxt = replay.ticks[t], replay.ticks[t + 1]
    true_state = state_from_tick(
        tick,
        dims=replay.dims,
        mountains=replay.mountains,
        castles=replay.cities,
        generals=replay.generals,
        time=t + time_phase,
    )
    per_player = []
    for player in range(len(replay.players)):
        events = extract_tick_actions(
            tick["owners"], tick["armies"], nxt["owners"], nxt["armies"],
            replay.cities, player, t,
        )
        per_player.append(candidate_action(events, replay.dims[0], replay.dims[1]))
    actions = jnp.asarray([list(per_player[0]), list(per_player[1])], dtype=jnp.int32)
    new_state, _ = game.step(true_state.engine_state, actions)
    h, w = replay.dims
    pred_armies = new_state.armies[:h, :w]
    pred_owner0 = new_state.ownership[0, :h, :w]
    pred_owner1 = new_state.ownership[1, :h, :w]
    true_armies = jnp.asarray(nxt["armies"], dtype=jnp.int32)
    true_owner0 = jnp.asarray([[o == 0 for o in row] for row in nxt["owners"]])
    true_owner1 = jnp.asarray([[o == 1 for o in row] for row in nxt["owners"]])
    armies_exact = bool(jnp.array_equal(pred_armies, true_armies))
    owners_exact = bool(jnp.array_equal(pred_owner0, true_owner0)) and bool(
        jnp.array_equal(pred_owner1, true_owner1)
    )
    army_cell_matches = int(jnp.sum(pred_armies == true_armies))
    total_cells = h * w
    return {
        "armies_exact": armies_exact,
        "owners_exact": owners_exact,
        "army_cell_parity": army_cell_matches / total_cells,
        "actions": per_player,
    }


def probe_replay(payload: dict, phases: list[int]) -> dict:
    replay = parse_replay(payload)
    n = len(replay.ticks) - 1
    per_phase = {}
    for phase in phases:
        exact = 0
        owners_ok = 0
        cell_parity_sum = 0.0
        for t in range(n):
            result = parity_at(replay, t, phase)
            exact += result["armies_exact"]
            owners_ok += result["owners_exact"]
            cell_parity_sum += result["army_cell_parity"]
        per_phase[str(phase)] = {
            "ticks": n,
            "armies_exact": exact,
            "owners_exact": owners_ok,
            "mean_cell_parity": round(cell_parity_sum / max(n, 1), 5),
        }
    return {"ticks": n, "dims": list(replay.dims), "phases": per_phase}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--phases", default="-2,-1,0,1,2")
    args = parser.parse_args()
    phases = [int(p) for p in args.phases.split(",")]
    raw_dir = Path(args.dataset_dir) / "raw"
    report = {
        "engine_sha": ENGINE_SUBMODULE_SHA,
        "ruleset": RULESET,
        "parser_version": "alignment-trace/1.0",
        "dataset_id": Path(args.dataset_dir).name,
        "replays": {},
        "aggregate": {},
    }
    for path in sorted(raw_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        report["replays"][path.name] = probe_replay(payload, phases)
    for phase in phases:
        total = sum(r["phases"][str(phase)]["armies_exact"] for r in report["replays"].values())
        ticks = sum(r["ticks"] for r in report["replays"].values())
        owners = sum(r["phases"][str(phase)]["owners_exact"] for r in report["replays"].values())
        report["aggregate"][str(phase)] = {
            "ticks": ticks,
            "armies_exact": total,
            "owners_exact": owners,
        }
        print(
            f"phase={phase:+d} armies_exact={total}/{ticks} owners_exact={owners}/{ticks}"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
