"""Build discovered-not-converted corpus and run conversion micro regression."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from generals_bot.evaluation.qualification_gates import evaluate_screening_smoke
from generals_bot.evaluation.qualification_runner import _play_qualification_game
from generals_bot.selector import create_policy

REPO = Path(__file__).resolve().parents[2]
MANIFEST_GLOB = "dev_development_*_heuristic_v2f_plus_planner.json"


FINE_CLASSES = (
    "IMMEDIATE_WIN_AVAILABLE_NOT_SELECTED",
    "IMMEDIATE_WIN_REJECTED_BY_RISK_GATE",
    "IMMEDIATE_WIN_REJECTED_BY_SHIELD",
    "GENERAL_HUNT_NOT_ACTIVATED",
    "GENERAL_HUNT_ACTIVATED_TOO_LATE",
    "SCOUT_ASSIGNMENT_NOT_CANCELLED",
    "ATTACK_STACK_NOT_REASSIGNED",
    "NO_REACHABLE_ROUTE_IN_KNOWN_MAP",
    "ROUTE_EXISTS_BUT_NOT_FOLLOWED",
    "ROUTE_RECOMPUTED_UNSTABLY",
    "OPTION_STICKINESS_BLOCKED_HUNT",
    "CONVENTIONAL_ARMY_MARGIN_VETO",
    "DEFENSIVE_CAUTION_BLOCKED_HUNT",
    "EMERGENCY_DEFENCE_BLOCKED_HUNT",
    "SOURCE_CATCHING_MODEL_OVERCONSERVATIVE",
    "GENERAL_LOCATION_MEMORY_LOST",
    "OTHER",
)


def build_corpus() -> list[dict]:
    rows: list[dict] = []
    for path in sorted((REPO / "experiments" / "manifests").glob(MANIFEST_GLOB)):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for g in doc.get("games", []):
            if not g.get("enemy_general_discovered"):
                continue
            if int(g.get("wins", 0)) != 0:
                continue
            rows.append(
                {
                    "replay_id": f"{g['policy']}_s{g['seed']}_p{g['position']}",
                    "source_manifest": path.name,
                    "policy": g["policy"],
                    "opponent": g["opponent"],
                    "seed": g["seed"],
                    "position": g["position"],
                    "terminal_turn": g["terminal_turn"],
                    "turn_enemy_general_discovered": g.get("turn_enemy_general_discovered"),
                    "failure_class": g.get("failure_class"),
                    "known_enemy_general": (g.get("extras") or {}).get("known_enemy_general"),
                    "candidate_config_hash": "57631a7d77ebd3f5",
                }
            )
    out = REPO / "experiments" / "manifests" / "phase_9q_discovered_not_converted_corpus.json"
    payload = {
        "schema_version": 1,
        "kind": "DISCOVERED_NOT_CONVERTED_CORPUS",
        "baseline_policy": "heuristic_v2f_plus_planner",
        "games": rows,
        "count": len(rows),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", out, "n=", len(rows))
    return rows


def run_micro(policy_name: str, corpus: list[dict]) -> dict:
    p = create_policy(policy_name)
    ch = getattr(p, "config_hash", None)
    results = []
    wins = 0
    for row in corpus:
        swap = bool(row["position"])
        g = _play_qualification_game(
            policy_name,
            row["opponent"],
            seed=int(row["seed"]),
            swap=swap,
            max_turns=1200,
        )
        won = int(g.wins) == 1
        wins += int(won)
        results.append(
            {
                "replay_id": row["replay_id"],
                "seed": row["seed"],
                "position": row["position"],
                "baseline_failure": row["failure_class"],
                "won": won,
                "terminal_turn": g.terminal_turn,
                "discovered": g.enemy_general_discovered,
                "discovery_turn": g.turn_enemy_general_discovered,
                "failure_class": g.failure_class,
                "extras": g.extras,
            }
        )
        print(
            f"  {row['replay_id']}: {'WIN' if won else 'FAIL'} "
            f"t={g.terminal_turn} disc@{g.turn_enemy_general_discovered} {g.failure_class}",
            flush=True,
        )
    n = len(corpus)
    report = {
        "schema_version": 1,
        "kind": "CONVERSION_MICRO_REGRESSION",
        "policy": policy_name,
        "config_hash": ch,
        "corpus_size": n,
        "wins": wins,
        "conversion_rate": wins / n if n else 0.0,
        "results": results,
        "gate": {
            "require_conversion_rate": 0.80,
            "passed": (wins / n if n else 0.0) >= 0.80,
        },
    }
    out = REPO / "experiments" / "manifests" / f"conversion_micro_{policy_name}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote", out, "conversion", report["conversion_rate"], "gate", report["gate"]["passed"])
    return report


def main() -> None:
    t0 = time.perf_counter()
    corpus = build_corpus()
    print("=== micro: terminal_fix ===", flush=True)
    r1 = run_micro("heuristic_v2f_plus_planner_terminal_fix", corpus)
    print("=== micro: baseline planner (control) ===", flush=True)
    r0 = run_micro("heuristic_v2f_plus_planner", corpus)
    summary = {
        "elapsed_s": time.perf_counter() - t0,
        "corpus": len(corpus),
        "baseline_conversion": r0["conversion_rate"],
        "terminal_fix_conversion": r1["conversion_rate"],
        "terminal_fix_gate": r1["gate"]["passed"],
    }
    path = REPO / "experiments" / "manifests" / "phase_9q_conversion_micro_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
