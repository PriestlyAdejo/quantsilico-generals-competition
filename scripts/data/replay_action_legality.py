"""Action legality audit for extracted elite-replay actions (BC-prep milestone).

ELITE_REPLAY_AUGMENTATION roadmap: action extraction + legality check before
the BC warm-start predeclaration. Extracted actions are heuristic estimates
(scripts/data/replay_action_extraction.py); before any of them may become a
behavioural-cloning training target, each MOVE must satisfy the generals.io
movement rules at its tick, checked ONLY against information the player could
legally hold (own armies/ownership are always legal knowledge; nothing hidden
is consulted):

- src owned by the player at tick t (extraction-time invariant);
- src army > 1 at tick t (a single army cannot move);
- moved amount in [1, armies_prev[src] - 1] (one army always stays);
- dst adjacent to src (replay tick granularity can defeat this - double moves
  and mid-tick captures make src/dst estimates non-adjacent; such events are
  UNUSABLE as BC targets and must be filtered, never repaired silently);
- dst not a mountain.

PASS/BUILD events are structural labels (BUILD = city capture) and are not
movement-constrained; they are counted separately and always consumable.

The audit emits a machine-readable report; consumers (BC sampling) must drop
every MOVE flagged illegal here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from scripts.data.replay_action_extraction import ActionEvent, extract_player_actions
from scripts.data.replay_legal_pov import Replay, parse_replay

MOVE_RULES = ("src_owned", "src_army_gt_1", "amount_in_range", "adjacent", "dst_not_mountain")


@dataclass
class MoveCheck:
    tick: int
    legal: bool
    violations: list[str]


def check_move(
    event: ActionEvent,
    owners_prev: list[list[int]],
    armies_prev: list[list[int]],
    mountains: set[tuple[int, int]],
    player: int,
) -> MoveCheck:
    violations: list[str] = []
    src, dst = event.src, event.dst
    if src is None or dst is None:
        return MoveCheck(event.tick, False, ["missing_endpoints"])
    if owners_prev[src[0]][src[1]] != player:
        violations.append("src_owned")
    src_army = armies_prev[src[0]][src[1]]
    if src_army <= 1:
        violations.append("src_army_gt_1")
    if not (1 <= event.amount <= src_army - 1):
        violations.append("amount_in_range")
    if abs(src[0] - dst[0]) + abs(src[1] - dst[1]) != 1:
        violations.append("adjacent")
    if dst in mountains:
        violations.append("dst_not_mountain")
    return MoveCheck(event.tick, not violations, violations)


def audit_replay(replay: Replay) -> dict:
    """Legality audit for every player in one replay."""
    per_player: dict[int, dict] = {}
    for player in range(len(replay.players)):
        events = extract_player_actions(replay, player)
        kinds = {"MOVE": 0, "BUILD": 0, "PASS": 0}
        legal_moves = 0
        violation_counts = {rule: 0 for rule in MOVE_RULES}
        for ev in events:
            kinds[ev.kind] = kinds.get(ev.kind, 0) + 1
            if ev.kind != "MOVE":
                continue
            tick = replay.ticks[ev.tick]
            result = check_move(ev, tick["owners"], tick["armies"], replay.mountains, player)
            if result.legal:
                legal_moves += 1
            for v in result.violations:
                if v in violation_counts:
                    violation_counts[v] += 1
        per_player[player] = {
            "events": kinds,
            "moves_legal": legal_moves,
            "moves_illegal": kinds["MOVE"] - legal_moves,
            "violation_counts": violation_counts,
        }
    return {
        "ticks": len(replay.ticks),
        "players": list(replay.players),
        "winner": replay.winner,
        "per_player": per_player,
    }


def audit_dataset(dataset_dir: Path) -> dict:
    """Audit every raw replay payload in a sealed dataset directory."""
    raw_dir = dataset_dir / "raw"
    reports: dict[str, dict] = {}
    totals = {"moves_legal": 0, "moves_illegal": 0, "replays": 0}
    for path in sorted(raw_dir.glob("*.json")):
        replay = parse_replay(json.loads(path.read_text(encoding="utf-8")))
        report = audit_replay(replay)
        reports[path.name] = report
        totals["replays"] += 1
        for player_report in report["per_player"].values():
            totals["moves_legal"] += player_report["moves_legal"]
            totals["moves_illegal"] += player_report["moves_illegal"]
    return {"dataset_id": dataset_dir.name, "totals": totals, "replays": reports}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out", default=None, help="write JSON report here")
    args = parser.parse_args()
    report = audit_dataset(Path(args.dataset_dir))
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    totals = report["totals"]
    print(
        f"dataset={report['dataset_id']} replays={totals['replays']} "
        f"moves_legal={totals['moves_legal']} moves_illegal={totals['moves_illegal']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
