"""STAGE5_TEACHER_R2 STEP1: hunter-vs-hunter SELF-PLAY teacher games.

Reuses the TEACHER-R1 recorded-transcript machinery (mirrors match.py;
deterministic replay verification). Gate is R2-specific and predeclared
(stage5_teacher_r2_plan.yaml): >= 16/20 games DECISIVE (a winner inside
the horizon; draws-at-truncation do not count), zero engine faults, all
transcripts replay-verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from scripts.data.teacher_r1_generate import HUNTER, play_recorded, verify_replay  # noqa: E402

OUT_ROOT = REPO / "experiments/marathon/teacher_r2/step1_selfplay"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="20260901-20260920")
    args = parser.parse_args()
    lo, hi = (int(x) for x in args.seeds.split("-"))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    decisive = engine_faults = 0
    docs = []
    for seed in range(lo, hi + 1):
        stamp = time.strftime("%H:%M:%SZ", time.gmtime())
        doc = play_recorded(seed, agent0=HUNTER, agent1=HUNTER)
        doc.update(verify_replay(doc))
        docs.append(doc)
        is_decisive = doc["winner"] in (0, 1) and not doc["truncated"]
        decisive += int(is_decisive)
        engine_faults += doc["faults_hunter"] + doc["faults_opponent"]
        print(
            f"{stamp} seed={seed} winner={doc['winner']} turns={doc['turns']} "
            f"truncated={doc['truncated']} decisive={is_decisive} "
            f"faults={doc['faults_hunter']}/{doc['faults_opponent']} "
            f"replay_match={doc['replay_match']} ({doc['elapsed_s']:.1f}s)",
            flush=True,
        )

    transcript_path = OUT_ROOT / "transcripts.json"
    payload = json.dumps(docs, sort_keys=True).encode()
    transcript_path.write_bytes(payload)
    summary = {
        "plan": "experiments/marathon/stage5_teacher_r2_plan.yaml",
        "experiment_id": "experiment#stage5-teacher-r2#8ab7def9c575",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "matchup": "official_hunter vs official_hunter (self-play, same policy both seats)",
        "seeds": list(range(lo, hi + 1)),
        "decisive_games": decisive,
        "games": len(docs),
        "engine_faults": engine_faults,
        "all_replays_match": all(d["replay_match"] for d in docs),
        "transcript_sha256": hashlib.sha256(payload).hexdigest(),
        "gate_predeclared": "decisive_games >= 16/20 AND engine_faults == 0 AND all_replays_match",
        "gate_pass": decisive >= 16 and engine_faults == 0
        and all(d["replay_match"] for d in docs),
    }
    (OUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, indent=1), flush=True)
    print("GATE_PASS" if summary["gate_pass"] else "GATE_FAIL", flush=True)
    return 0 if summary["gate_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
