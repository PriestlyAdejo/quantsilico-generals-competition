"""BC-A-FULL-R1 union shard builder (predeclared: bc_a_full_round_1_plan.yaml).

Union of the four elite datasets (EV-0052), deduped by match id across
snapshots. Label/split policy inherits the pilot builder (EV-0050) with two
predeclared changes:

  * holdout players: ResBot PLUS the single highest-volume remaining player
    by match count, chosen HERE (build time, before training, recorded in the
    manifest - never post-hoc);
  * phase-stratified labels: every sample carries its turn-phase bucket.

Labels come ONLY from engine-verified derivation (EXACT_MATCH + OWNERS_ONLY;
NO_MATCH excluded; ENGINE_SILENT_PASS labels excluded). Observations are
never stored; the trainer reconstructs them through the canonical legal
observation path. PPO_SEMANTICS: OFF_POLICY_AUXILIARY.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src", REPO / "third_party" / "generals-bots"):
    entry = str(candidate)
    if entry not in sys.path:
        sys.path.insert(0, entry)

from scripts.data.bc_a_build_shards import (  # noqa: E402
    classify_label_action,
    engine_action_to_index,
    replay_sha256,
)
from scripts.data.replay_action_derivation import derive_tick  # noqa: E402
from scripts.data.replay_engine_oracle import (  # noqa: E402
    ENGINE_SILENT_PASS,
    ENGINE_SUBMODULE_SHA,
    RULESET,
    state_from_tick,
)
from scripts.data.replay_legal_pov import parse_replay  # noqa: E402

DATASET_DIRS = [
    "experiments/datasets/elite_replays/DATASET-ELITE-2026-08-15-V01",
    "experiments/datasets/elite_replays/DATASET-ELITE-2026-08-16-V01",
    "experiments/datasets/elite_replays/DATASET-ELITE-2026-08-16-V02",
    "experiments/datasets/elite_replays/DATASET-ELITE-2026-08-16-V03",
]
FIRST_HOLDOUT = "ResBot"
PHASES = ((0, 200, "0-199"), (200, 400, "200-399"), (400, 800, "400-799"), (800, 10**9, "800+"))


def phase_of(tick: int) -> str:
    for lo, hi, name in PHASES:
        if lo <= tick < hi:
            return name
    return "800+"


def collect_union() -> tuple[list[tuple[Path, dict]], int]:
    """Union of raw replay paths, deduped by match id; returns (paths, dupes)."""
    seen: dict[str, tuple[Path, dict]] = {}
    dupes = 0
    for dataset_dir in DATASET_DIRS:
        root = REPO / dataset_dir
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        by_id = {m["id"]: m for m in manifest["matches"]}
        for path in sorted((root / "raw").glob("*.json")):
            meta = by_id.get(path.stem)
            if meta is None:
                raise SystemExit(f"raw replay {path} missing from its manifest")
            if meta["id"] in seen:
                dupes += 1
                continue
            seen[meta["id"]] = (path, meta)
    return list(seen.values()), dupes


def choose_second_holdout(match_counts: dict[str, int]) -> str:
    remaining = {name: n for name, n in match_counts.items() if name != FIRST_HOLDOUT}
    best = max(remaining.items(), key=lambda kv: (kv[1], kv[0]))
    return best[0]


def build(out_dir: Path) -> dict:
    entries, dupes = collect_union()
    if len(entries) < 81:
        raise SystemExit(f"union has {len(entries)} games; predeclared >= 81")

    match_counts: dict[str, int] = {}
    for _, meta in entries:
        for name in (meta["a_name"], meta["b_name"]):
            match_counts[name] = match_counts.get(name, 0) + 1
    second_holdout = choose_second_holdout(match_counts)
    holdout_players = {FIRST_HOLDOUT, second_holdout}

    # time-disjoint: each TRAINING player's longest replay is held out
    per_player: dict[str, list[tuple[Path, dict]]] = {}
    for path, meta in entries:
        for name in (meta["a_name"], meta["b_name"]):
            if name not in holdout_players:
                per_player.setdefault(name, []).append((path, meta))
    time_holdouts: set[Path] = {
        sorted(plist, key=lambda e: int(e[1]["turns"]))[-1][0] for plist in per_player.values()
    }

    samples: list[dict] = []
    counters = {
        "ticks_derived": 0,
        "NO_MATCH_excluded": 0,
        "silent_pass_excluded": 0,
        "general_move_flagged": 0,
        "labels_kept": 0,
    }
    split_counts = {"train": 0, "holdout_player": 0, "holdout_replay": 0}
    phase_counts = {name: 0 for _, _, name in PHASES}
    provenance_hashes: dict[str, str] = {}

    for index, (path, _meta) in enumerate(entries):
        payload = json.loads(path.read_text(encoding="utf-8"))
        replay = parse_replay(payload)
        provenance_hashes[f"{path.parent.parent.name}/{path.name}"] = replay_sha256(path)
        is_time_holdout = path in time_holdouts
        for t in range(len(replay.ticks) - 1):
            result = derive_tick(replay, t)
            counters["ticks_derived"] += 1
            if result["status"] == "NO_MATCH":
                counters["NO_MATCH_excluded"] += 1
                continue
            state = state_from_tick(
                replay.ticks[t],
                dims=replay.dims,
                mountains=replay.mountains,
                castles=replay.cities,
                generals=replay.generals,
                time=t,
            )
            for seat in range(2):
                player_name = replay.players[seat]
                if player_name in holdout_players:
                    split = "holdout_player"
                elif is_time_holdout:
                    split = "holdout_replay"
                else:
                    split = "train"
                action = tuple(result["actions"][seat])
                outcome = classify_label_action(state, seat, action)
                if outcome == ENGINE_SILENT_PASS:
                    counters["silent_pass_excluded"] += 1
                    continue
                general_flag = (
                    result["status"] == "OWNERS_ONLY"
                    and action[0] == 0
                    and replay.generals.get(seat) == (action[1], action[2])
                )
                if general_flag:
                    counters["general_move_flagged"] += 1
                phase = phase_of(t)
                samples.append(
                    {
                        "replay": path.name,
                        "dataset": path.parent.parent.name,
                        "tick": t,
                        "seat": seat,
                        "player": player_name,
                        "split": split,
                        "phase": phase,
                        "label": engine_action_to_index(action),
                        "derivation_status": result["status"],
                        "general_move_flag": bool(general_flag),
                    }
                )
                counters["labels_kept"] += 1
                split_counts[split] += 1
                phase_counts[phase] += 1
        if (index + 1) % 5 == 0:
            print(
                f"progress: {index + 1}/{len(entries)} replays, "
                f"labels_kept={counters['labels_kept']}",
                flush=True,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
    shard_manifest = {
        "kind": "MARATHON_BC_DERIVED_SHARD_MANIFEST",
        "dataset_id": "BC-DERIVED-UNION-A2",
        "source_datasets": [Path(d).name for d in DATASET_DIRS],
        "union_games": len(entries),
        "cross_snapshot_dupes_dropped": dupes,
        "match_counts": match_counts,
        "holdout_players": sorted(holdout_players),
        "holdout_selection_rule": (
            f"{FIRST_HOLDOUT} fixed by predeclaration; {second_holdout} = single "
            "highest-volume remaining player by match count, chosen at build "
            f"time ({match_counts[second_holdout]} matches), before training"
        ),
        "source_replay_hashes": provenance_hashes,
        "samples_sha256": replay_sha256(samples_path),
        "engine_sha": ENGINE_SUBMODULE_SHA,
        "ruleset": RULESET,
        "derivation_version": "exact-derivation/1.0 (EV-0045)",
        "builder": "bc_a_build_shards_full/1.0",
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label_policy": (
            "EXACT_MATCH + OWNERS_ONLY engine-verified action pairs only; "
            "NO_MATCH excluded; ENGINE_SILENT_PASS labels excluded; "
            "source-site general-move ticks FLAGGED (general_move_flag)"
        ),
        "observation_policy": (
            "features reconstructed at training time through the canonical legal "
            "observation path (state_from_tick + observe_one_jax with tracked fog "
            "memory); full hidden state never enters features; leak-guard fixture "
            "tests gate the trainer"
        ),
        "splits": {
            "time_disjoint_rule": "each training player's longest replay held out",
            "counts": split_counts,
        },
        "phase_counts": phase_counts,
        "counters": counters,
        "consumption_gate": (
            "a passing checkpoint is eligible ONLY as warm-start / auxiliary input "
            "for a separately predeclared PPO-continuation sub-experiment; BC "
            "accuracy never promotes (plan consumption_gate)"
        ),
    }
    (out_dir / "shards.json").write_text(
        json.dumps(shard_manifest, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"counters": counters, "splits": split_counts}, indent=1))
    return shard_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    build(Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
