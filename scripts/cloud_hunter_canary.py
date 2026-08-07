#!/usr/bin/env python3
"""Run a tiny, CPU-only, seat-swapped Hunter canary for one COMPLETE checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "wins": sum(row.get("wdl") == "win" for row in results),
        "draws": sum(row.get("wdl") == "draw" for row in results),
        "losses": sum(row.get("wdl") == "loss" for row in results),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_competition_native_jax_daytime_eval import (
        OPPONENT_ALIASES,
        CNJPolicyAdapter,
        load_cnj_from_ckpt,
        play_game,
    )

    from generals_bot.selector import create_policy

    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "COMPLETE").is_file():
        raise RuntimeError(f"checkpoint is not COMPLETE: {checkpoint}")
    meta = json.loads((checkpoint / "meta.json").read_text(encoding="utf-8"))
    policy = load_cnj_from_ckpt(checkpoint, which="ema")
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for seat in (0, 1):
        opponent = create_policy(OPPONENT_ALIASES["official_hunter"], seed=args.seed)
        candidate = CNJPolicyAdapter(policy.inner.weights, seed=args.seed + seat)
        player0, player1 = (candidate, opponent) if seat == 0 else (opponent, candidate)
        game_started = time.perf_counter()
        result = play_game(
            player0,
            player1,
            seed=args.seed,
            max_turns=args.max_turns,
            focal_seat=seat,
        )
        result.update(
            {
                "opponent": "official_hunter",
                "seed": args.seed,
                "focal_seat": seat,
                "elapsed_s": time.perf_counter() - game_started,
            }
        )
        results.append(result)
        print(f"seat={seat} wdl={result.get('wdl')} turns={result.get('turns')}", flush=True)
    report = {
        "schema_version": 1,
        "kind": "CLOUD_HUNTER_CANARY",
        "status": "COMPLETE",
        "checkpoint": str(checkpoint),
        "checkpoint_transitions": int(meta["transitions"]),
        "checkpoint_update": int(meta["update"]),
        "ema_sha256": sha256(checkpoint / "ema.npz"),
        "protocol": {
            "opponent": "official_hunter",
            "seed": args.seed,
            "seat_swaps": True,
            "games": 2,
            "max_turns": args.max_turns,
            "backend": "cpu",
        },
        "summary": summarize(results),
        "results": results,
        "elapsed_s": time.perf_counter() - started,
        "written_at": datetime.now(UTC).isoformat(),
        "selection_warning": "Two-game canary only; not sufficient for teacher selection.",
    }
    atomic_json(args.output, report)
    print(json.dumps({"summary": report["summary"], "output": str(args.output)}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--max-turns", type=int, default=1200)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
