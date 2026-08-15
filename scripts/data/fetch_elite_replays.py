"""Bounded elite-replay snapshot fetcher (ELITE_REPLAY_AUGMENTATION §2/§3).

Independent implementation from the observed public API contract
(https://www.generals.bot/api/leaderboard; concept per operator gist
mrinmoy2developer/fetch_top_replay.py — not vendored):

- leaderboard listing -> top non-baseline users (or explicit names)
- per-player match listing (?matches=1&player=NAME)
- seat-order dedup by (seed, sorted player pair)
- full state timeline download (?replay=ID), bounded workers, 403 backoff
- resumable; raw payloads immutable; manifest + sha256 hashes + dataset id

Dataset identity: DATASET-ELITE-YYYY-MM-DD-VNN allocated under the out root;
a snapshot is never overwritten because the leaderboard moved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

API = "https://www.generals.bot/api/leaderboard"
UA = "qs-marathon-replay-pilot/1"
FETCHER_VERSION = "fetch_elite_replays/1.0"


def get_json(url: str, retries: int = 4):
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": UA})
            with urlopen(req, timeout=60) as fh:  # noqa: S310 - fixed public API
                return json.loads(fh.read())
        except HTTPError as exc:
            last = exc
            # Endpoint burst-limit responds 403, not 429: back off harder.
            delay = min(60.0, 8.0 * (2**attempt)) if exc.code == 403 else 2.0 * (attempt + 1)
            time.sleep(delay)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"GET failed: {url}: {last}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_matches(players: list[str], per_player: int, *, min_turns: int):
    chosen: list[dict] = []
    seen: set[tuple[int, tuple[str, ...]]] = set()
    by_player: dict[str, int] = {p: 0 for p in players}
    for player in players:
        listing = get_json(f"{API}?matches=1&player={quote(player)}")
        for match in listing.get("matches", []):
            names = tuple(n for n in (match.get("a_name"), match.get("b_name")) if n)
            if player not in names:
                continue
            if int(match.get("turns", 0) or 0) < min_turns:
                continue
            key = (int(match.get("seed", 0)), tuple(sorted(names)))
            if key in seen:
                continue
            seen.add(key)
            chosen.append(match)
            by_player[player] += 1
            if by_player[player] >= per_player:
                break
    return chosen, by_player


def allocate_dataset_dir(root: Path, date_str: str) -> Path:
    """New version per sealed snapshot; an incomplete snapshot is resumed."""
    version = 1
    while True:
        candidate = root / f"DATASET-ELITE-{date_str}-V{version:02d}"
        if not candidate.exists():
            return candidate
        if not (candidate / "manifest.json").exists():
            return candidate  # incomplete fetch: resume in place
        version += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--players", default=None, help="comma-separated names; overrides --top")
    parser.add_argument("--per-player", type=int, default=2)
    parser.add_argument("--min-turns", type=int, default=40)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("experiments/datasets/elite_replays"))
    args = parser.parse_args()

    board = get_json(API)
    if args.players:
        players = [p.strip() for p in args.players.split(",") if p.strip()]
    else:
        players = [x["label"] for x in board.get("leaderboard", []) if x.get("kind") == "user"][
            : args.top
        ]
    if not players:
        print("leaderboard returned no users", file=sys.stderr)
        return 1

    matches, counts = select_matches(players, args.per_player, min_turns=args.min_turns)
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    root = allocate_dataset_dir(args.out, date_str)
    dataset_id = root.name
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    pending = []
    done = 0
    for match in matches:
        rid = str(match["id"])
        path = raw / f"{rid}.json"
        if path.exists() and path.stat().st_size > 100:
            done += 1
            continue
        pending.append((rid, path))
    print(f"selected {len(matches)} distinct games; present {done}; fetching {len(pending)}")

    errors: list[str] = []

    def download_one(item):
        rid, path = item
        payload = get_json(f"{API}?replay={quote(rid)}")
        body = json.dumps(payload, separators=(",", ":"))
        tmp = path.with_suffix(".json.part")
        tmp.write_text(body)
        tmp.replace(path)  # atomic-ish: never leave a torn raw payload
        return rid, len(payload.get("ticks", []))

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(download_one, item) for item in pending]
        for fut in as_completed(futures):
            try:
                rid, ticks = fut.result()
                done += 1
                print(f"{done}/{len(matches)}  replay {rid}  {ticks} ticks", flush=True)
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
                print(f"download error ({len(errors)}): {exc}", flush=True)

    payloads = sorted(raw.glob("*.json"))
    hashes = {p.name: sha256_file(p) for p in payloads}
    manifest = {
        "kind": "MARATHON_ELITE_REPLAY_SNAPSHOT_MANIFEST",
        "dataset_id": dataset_id,
        "fetched_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_endpoint": API,
        "source_implementation": FETCHER_VERSION,
        "source_concept_gist": "mrinmoy2developer/c5ee19453a77947d7941deb61318f5fb",
        "filtering": {
            "per_player": args.per_player,
            "min_turns": args.min_turns,
            "dedup_policy": "seed+sorted_player_pair (collapses seat-order duplicates)",
        },
        "players": players,
        "per_player_counts": counts,
        "leaderboard_sha256": hashlib.sha256(
            json.dumps(board, sort_keys=True).encode()
        ).hexdigest(),
        "matches": matches,
        "payload_hashes": hashes,
        "observation_reconstruction_version": None,  # filled by downstream legal-POV stage
        "action_extraction_version": None,
        "split": None,
        "content_hash": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if errors:
        print(f"{len(errors)} downloads failed; rerun to resume", file=sys.stderr)
        return 2
    print(json.dumps({"dataset_id": dataset_id, "replays": len(payloads), "root": str(root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
