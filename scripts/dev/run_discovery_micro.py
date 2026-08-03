"""Freeze no-discovery draws and measure discovery after scout fixes."""

from __future__ import annotations

import json
import time
from pathlib import Path

from generals_bot.evaluation.qualification_runner import _play_qualification_game
from generals_bot.selector import create_policy

REPO = Path(__file__).resolve().parents[2]
DEV_GLOB = "dev_development_*_heuristic_v2f_plus_planner_terminal_fix.json"


def build_no_discovery_corpus() -> list[dict]:
    rows: list[dict] = []
    for path in sorted((REPO / "experiments" / "manifests").glob(DEV_GLOB)):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for g in doc.get("games", []):
            if g.get("enemy_general_discovered"):
                continue
            rows.append(
                {
                    "replay_id": f"{g['policy']}_s{g['seed']}_p{g['position']}",
                    "source_manifest": path.name,
                    "seed": g["seed"],
                    "position": g["position"],
                    "opponent": g["opponent"],
                    "failure_class": g.get("failure_class"),
                    "terminal_turn": g.get("terminal_turn"),
                    "last_newly_scouted_turn": g.get("last_newly_scouted_turn"),
                    "first_enemy_contact_turn": g.get("first_enemy_contact_turn"),
                    "candidate_general_cells_terminal": g.get("candidate_general_cells_terminal"),
                    "remaining_enemy_land": g.get("remaining_enemy_land"),
                    "land_ratio_terminal": g.get("land_ratio_terminal"),
                    "prior_extras": {
                        k: (g.get("extras") or {}).get(k)
                        for k in (
                            "scout_source",
                            "scout_target",
                            "scout_stall",
                            "scout_abort_reason",
                            "candidate_mask_size",
                            "phase",
                        )
                    },
                }
            )
    out = REPO / "experiments" / "manifests" / "phase_9q_no_discovery_corpus.json"
    payload = {
        "schema_version": 1,
        "kind": "NO_DISCOVERY_CORPUS",
        "baseline_policy": "heuristic_v2f_plus_planner_terminal_fix",
        "baseline_config_hash": "4c5466776180217b",
        "label": "PHASE_9Q_NO_DISCOVERY_AFTER_CONVERSION_FIX",
        "games": rows,
        "count": len(rows),
        "failure_classes": {},
    }
    from collections import Counter

    payload["failure_classes"] = dict(Counter(r["failure_class"] for r in rows))
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", out, "n=", len(rows), payload["failure_classes"])
    return rows


def run_discovery_micro(policy_name: str, corpus: list[dict]) -> dict:
    p = create_policy(policy_name)
    ch = getattr(p, "config_hash", None)
    results = []
    discovered = 0
    wins = 0
    t0 = time.perf_counter()
    for row in corpus:
        swap = bool(int(row["position"]))
        rec = _play_qualification_game(
            policy_name,
            row["opponent"],
            seed=int(row["seed"]),
            swap=swap,
            max_turns=1200,
        )
        disc = bool(rec.enemy_general_discovered)
        win = int(rec.wins) == 1
        if disc:
            discovered += 1
        if win:
            wins += 1
        results.append(
            {
                "replay_id": row["replay_id"],
                "seed": row["seed"],
                "position": row["position"],
                "prior_failure": row["failure_class"],
                "discovered": disc,
                "discovery_turn": rec.turn_enemy_general_discovered,
                "wins": rec.wins,
                "draws": rec.draws,
                "losses": rec.losses,
                "failure_class": rec.failure_class,
                "last_newly_scouted_turn": rec.last_newly_scouted_turn,
            }
        )
        print(
            f"  seed={row['seed']} p={row['position']} disc={disc} "
            f"W={rec.wins} fail={rec.failure_class}",
            flush=True,
        )
    n = len(corpus)
    out = {
        "schema_version": 1,
        "kind": "PHASE_9Q_DISCOVERY_MICRO",
        "policy": policy_name,
        "config_hash": ch,
        "corpus_size": n,
        "discovered": discovered,
        "discovery_rate": discovered / n if n else 0.0,
        "wins": wins,
        "win_rate": wins / n if n else 0.0,
        "elapsed_s": time.perf_counter() - t0,
        "results": results,
        "gate": {
            "passed": (discovered / n if n else 0.0) >= 0.30,
            "require_discovery_rate_ge": 0.30,
            "note": "Micro gate on previously no-discovery draws only; not development gate.",
        },
    }
    path = REPO / "experiments" / "manifests" / f"discovery_micro_{policy_name}.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"discovery_micro {policy_name}: {discovered}/{n} "
        f"gate={'PASS' if out['gate']['passed'] else 'FAIL'}",
        flush=True,
    )
    print("wrote", path, flush=True)
    return out


def main() -> None:
    corpus = build_no_discovery_corpus()
    run_discovery_micro("heuristic_v2f_plus_planner_terminal_fix", corpus)


if __name__ == "__main__":
    main()
