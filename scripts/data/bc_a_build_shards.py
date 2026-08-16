"""BC-A pilot shard builder (predeclared: bc_a_pilot_round_1_plan.yaml).

Builds BC training/eval shards from DATASET-ELITE under the competition
authority (EV-0042/0045/0037):

  * labels come ONLY from engine-verified derivation (derive_tick):
    EXACT_MATCH + OWNERS_ONLY ticks; NO_MATCH excluded;
  * derived label actions that the pinned engine classifies as
    ENGINE_SILENT_PASS at their tick are EXCLUDED (predeclared rule);
  * ticks with the source-site general-move army residual (EV-0045) are
    FLAGGED in the manifest (labels unaffected - ownership-verified);
  * splits: player-disjoint (ResBot holdout) + time-disjoint (each training
    player's LAST replay held out); no row-random splits;
  * observations are never stored: the trainer reconstructs them through
    the canonical legal observation path (state_from_tick + observe_one_jax
    with per-replay fog memory); the shard stores replay/tick/seat + label.

Output: <out>/shards.json (manifest, DERIVED dataset identity + provenance)
and <out>/samples.jsonl. PPO_SEMANTICS: OFF_POLICY_AUXILIARY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src", REPO / "third_party" / "generals-bots"):
    entry = str(candidate)
    if entry not in sys.path:
        sys.path.insert(0, entry)

from generals.core.game import DIRECTIONS  # noqa: E402

from scripts.data.replay_action_derivation import derive_tick  # noqa: E402
from scripts.data.replay_engine_oracle import (  # noqa: E402
    ENGINE_SILENT_PASS,
    ENGINE_SUBMODULE_SHA,
    RULESET,
    classify_build,
    classify_move,
    state_from_tick,
)
from scripts.data.replay_legal_pov import parse_replay  # noqa: E402

HOLDOUT_PLAYER = "ResBot"
MAX_HW = 21


def engine_action_to_index(action: tuple[int, int, int, int, int]) -> int:
    kind, row, col, direction, split = action
    if kind == 1:
        return 0
    cell = row * MAX_HW + col
    if kind == 2:
        return 1 + cell * 9 + 8
    return 1 + cell * 9 + direction * 2 + split


def classify_label_action(state, seat: int, action: tuple[int, int, int, int, int]) -> str:
    kind, row, col, direction, split = action
    if kind == 1:
        return "ENGINE_EXECUTED"  # pass always executes
    if kind == 2:
        return classify_build(state, seat, (row, col)).engine_outcome
    dst = (row + int(DIRECTIONS[direction, 0]), col + int(DIRECTIONS[direction, 1]))
    return classify_move(state, seat, (row, col), dst, split).engine_outcome


def replay_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(dataset_dir: Path, out_dir: Path) -> dict:
    raw_dir = dataset_dir / "raw"
    paths = sorted(raw_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"no raw replays under {raw_dir}")
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    by_id = {m["id"]: m for m in manifest["matches"]}

    # Split assignment (predeclared)
    per_player_replays: dict[str, list[Path]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for player in payload.get("players", []):
            per_player_replays.setdefault(player, []).append(path)
    time_holdouts: set[Path] = set()
    for player, plist in per_player_replays.items():
        if player != HOLDOUT_PLAYER:
            time_holdouts.add(sorted(plist, key=lambda p: int(by_id[p.stem]["turns"]))[-1])

    samples: list[dict] = []
    counters = {
        "ticks_derived": 0,
        "NO_MATCH_excluded": 0,
        "silent_pass_excluded": 0,
        "general_move_flagged": 0,
        "labels_kept": 0,
    }
    split_counts = {"train": 0, "holdout_player": 0, "holdout_replay": 0}
    provenance_hashes = {}

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        replay = parse_replay(payload)
        provenance_hashes[path.name] = replay_sha256(path)
        is_holdout_player_replay = HOLDOUT_PLAYER in replay.players
        is_time_holdout = path in time_holdouts
        for t in range(len(replay.ticks) - 1):
            result = derive_tick(replay, t)
            counters["ticks_derived"] += 1
            if result["status"] == "NO_MATCH":
                counters["NO_MATCH_excluded"] += 1
                continue
            tick = replay.ticks[t]
            state = state_from_tick(
                tick,
                dims=replay.dims,
                mountains=replay.mountains,
                castles=replay.cities,
                generals=replay.generals,
                time=t,
            )
            for seat in range(2):
                player_name = replay.players[seat]
                if is_holdout_player_replay:
                    if player_name != HOLDOUT_PLAYER:
                        continue  # label seat must be the holdout player
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
                label_idx = engine_action_to_index(action)
                general_flag = (
                    result["status"] == "OWNERS_ONLY"
                    and action[0] == 0
                    and replay.generals.get(seat) == (action[1], action[2])
                )
                if general_flag:
                    counters["general_move_flagged"] += 1
                samples.append(
                    {
                        "replay": path.name,
                        "tick": t,
                        "seat": seat,
                        "player": player_name,
                        "split": split,
                        "label": label_idx,
                        "derivation_status": result["status"],
                        "general_move_flag": bool(general_flag),
                    }
                )
                counters["labels_kept"] += 1
                split_counts[split] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
    samples_hash = replay_sha256(samples_path)
    shard_manifest = {
        "kind": "MARATHON_BC_DERIVED_SHARD_MANIFEST",
        "dataset_id": f"BC-DERIVED-{manifest['dataset_id']}-A1",
        "source_dataset": manifest["dataset_id"],
        "source_replay_hashes": provenance_hashes,
        "samples_sha256": samples_hash,
        "engine_sha": ENGINE_SUBMODULE_SHA,
        "ruleset": RULESET,
        "derivation_version": "exact-derivation/1.0 (EV-0045)",
        "builder": "bc_a_build_shards/1.0",
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
            "holdout_player": HOLDOUT_PLAYER,
            "time_disjoint_rule": "each training player's longest replay held out",
            "counts": split_counts,
        },
        "counters": counters,
        "consumption_gate": (
            "pilot checkpoint NEVER enters the training funnel or gameplay eval; "
            "BC accuracy never promotes (plan consumption_gate)"
        ),
    }
    (out_dir / "shards.json").write_text(
        json.dumps(shard_manifest, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"counters": counters, "splits": split_counts}, indent=1))
    return shard_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    build(Path(args.dataset_dir), Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
