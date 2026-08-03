"""Diagnose behaviour-cloning train/validation generalisation gap."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from generals_bot.models.action_index import ACTION_DIM, BUILD_OFFSET, MOVE_OFFSET, PASS_INDEX
from generals_bot.models.heads import STRATEGIC_OPTIONS


def _action_kind(idx: int) -> str:
    if idx == PASS_INDEX:
        return "pass"
    if BUILD_OFFSET <= idx < MOVE_OFFSET:
        return "build"
    if MOVE_OFFSET <= idx < ACTION_DIM:
        return "move"
    return "invalid"


def analyse_dataset(train_path: Path, val_path: Path) -> dict:
    train = np.load(train_path, allow_pickle=True)
    val = np.load(val_path, allow_pickle=True)

    def slice_stats(data, name: str) -> dict:
        actions = data["action_index"]
        options = data["option_index"]
        sources = data["source"] if "source" in data.files else np.array(["unknown"] * len(actions))
        kinds = Counter(_action_kind(int(a)) for a in actions)
        opt_counts = Counter(STRATEGIC_OPTIONS[int(o)] if int(o) < len(STRATEGIC_OPTIONS) else str(o) for o in options)
        src_counts = Counter(str(s) for s in sources)
        # Hash cells for near-duplicates
        cell_hashes = [hash(c.tobytes()) for c in data["cells"]]
        dup_rate = 1.0 - (len(set(cell_hashes)) / max(len(cell_hashes), 1))
        return {
            "n": int(len(actions)),
            "unique_actions": int(len(set(map(int, actions)))),
            "action_kinds": dict(kinds),
            "options": dict(opt_counts),
            "sources": dict(src_counts),
            "duplicate_cell_rate": dup_rate,
            "pass_rate": kinds.get("pass", 0) / max(len(actions), 1),
            "build_rate": kinds.get("build", 0) / max(len(actions), 1),
            "move_rate": kinds.get("move", 0) / max(len(actions), 1),
        }

    train_s = slice_stats(train, "train")
    val_s = slice_stats(val, "val")

    # Overlap of action indices
    train_acts = set(map(int, train["action_index"]))
    val_acts = set(map(int, val["action_index"]))
    overlap = train_acts & val_acts
    val_only = val_acts - train_acts

    # Source/option shift
    def dist_shift(a: dict, b: dict) -> dict:
        keys = sorted(set(a) | set(b))
        out = {}
        na = sum(a.values()) or 1
        nb = sum(b.values()) or 1
        for k in keys:
            out[k] = {"train_pct": a.get(k, 0) / na, "val_pct": b.get(k, 0) / nb}
        return out

    report = {
        "schema_version": 1,
        "train": train_s,
        "validation": val_s,
        "action_index_overlap": len(overlap),
        "validation_actions_unseen_in_train": len(val_only),
        "validation_unseen_action_fraction": len(val_only) / max(len(val_acts), 1),
        "option_shift": dist_shift(train_s["options"], val_s["options"]),
        "source_shift": dist_shift(train_s["sources"], val_s["sources"]),
        "findings": [],
        "recommendations": [],
    }

    if report["validation_unseen_action_fraction"] > 0.5:
        report["findings"].append(
            "Majority of validation action indices never appear in training — "
            "flat ACTION_DIM imitation cannot generalise those exact moves."
        )
        report["recommendations"].append(
            "Use hierarchical action heads (source/direction/split) or legal-masked "
            "imitation with more seeds; avoid requiring exact flat-index match."
        )
    if abs(train_s["pass_rate"] - val_s["pass_rate"]) > 0.15:
        report["findings"].append("Large pass-rate shift between train and validation.")
    if train_s["duplicate_cell_rate"] > 0.2:
        report["findings"].append("High duplicate observation rate in training.")
        report["recommendations"].append("Deduplicate trajectories / diversify seeds.")
    src_train = train_s["sources"]
    if src_train and max(src_train.values()) / train_s["n"] > 0.5:
        report["findings"].append("Training dominated by one teacher source.")
        report["recommendations"].append("Balance heuristic teachers in collection.")
    report["findings"].append(
        "IID frame sampling with zeroed recurrent state ignores sequence boundaries; "
        "validation maps/seeds differ so exact action indices rarely match."
    )
    report["recommendations"].append(
        "Report legal-top1 and option accuracy alongside flat action accuracy; "
        "expand train seeds; keep promotion_holdout untouched."
    )

    path = Path("experiments/manifests/bc_generalisation_report.json")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["path"] = str(path)
    return report


def main() -> None:
    report = analyse_dataset(
        Path("experiments/datasets/bc/smoke_train.npz"),
        Path("experiments/datasets/bc/smoke_val.npz"),
    )
    print(json.dumps({k: report[k] for k in report if k not in {"option_shift", "source_shift"}}, indent=2))


if __name__ == "__main__":
    main()
