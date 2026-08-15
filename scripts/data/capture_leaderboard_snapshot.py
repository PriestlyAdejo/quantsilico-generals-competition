"""Timestamped public leaderboard snapshot (ELITE_REPLAY_AUGMENTATION §9).

Persists exactly what the public API exposes — rating/rank/match counters as
exposed, nothing fabricated. Snapshots append to a JSONL telemetry feed; each
line carries a capture timestamp and a content hash so trajectories can be
derived later (rating slope, rank velocity) where observations suffice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

API = "https://www.generals.bot/api/leaderboard"
UA = "qs-marathon-replay-pilot/1"


def fetch_board() -> dict:
    req = Request(API, headers={"User-Agent": UA})
    with urlopen(req, timeout=60) as fh:  # noqa: S310 - fixed public API
        return json.loads(fh.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/datasets/leaderboard_snapshots/snapshots.jsonl"),
    )
    args = parser.parse_args()

    board = fetch_board()
    entries = []
    for rank, row in enumerate(board.get("leaderboard", []), start=1):
        # Copy only scalar fields the API actually exposes.
        entry = {k: v for k, v in row.items() if isinstance(v, (str, int, float, bool, type(None)))}
        entry["rank_observed"] = rank
        entries.append(entry)
    record = {
        "kind": "LEADERBOARD_SNAPSHOT",
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_endpoint": API,
        "entry_count": len(entries),
        "content_hash": hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(),
        "entries": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    print(
        json.dumps(
            {
                "captured": record["captured_at_utc"],
                "entries": record["entry_count"],
                "hash": record["content_hash"][:16],
                "out": str(args.out),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
