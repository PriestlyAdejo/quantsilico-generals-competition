"""Phase 9Q controlled ablation matrix on fixed smoke seeds."""

from __future__ import annotations

import json
import time
from pathlib import Path

from generals_bot.evaluation.qualification_runner import run_qualification_suite
from generals_bot.policies.heuristic_v2_ablations import FLAGS
from generals_bot.selector import create_policy

REPO_ROOT = Path(__file__).resolve().parents[2]

# Ablation matrix A–I (plus references)
CANDIDATES = [
    "heuristic_v1_reference",
    "heuristic_v2f_best_reference",
    "heuristic_v2_9qd_latest",
    "heuristic_v2f_restored",
    "heuristic_v2f_plus_planner",
    "heuristic_v2f_plus_garrison",
    "heuristic_v2f_plus_hunter_intercept",
    "heuristic_v2f_plus_planner_plus_garrison",
    "heuristic_v2f_plus_planner_plus_intercept",
    "heuristic_v2f_plus_planner_plus_garrison_plus_intercept",
]


def _config_hash(name: str) -> str:
    try:
        p = create_policy(name)
        if hasattr(p, "config_hash"):
            return str(p.config_hash)
        if name in FLAGS:
            return FLAGS[name].config_hash()
        if "v2f" in name:
            return "frozen_544215e_v2f"
        if "9qd" in name or name.endswith("qualifier"):
            return "9f60ffa_v2_qualifier"
        if "v1" in name:
            return "heuristic_v1"
    except Exception as exc:  # noqa: BLE001
        return f"error:{exc}"
    return "unknown"


def run_matrix(*, wall_clock_s: float | None = 7200.0) -> dict:
    from dataclasses import replace
    from generals_bot.evaluation import qualification_runner as qr

    # Expander smoke: 8 seeds paired = 16 games
    # Hunter micro: 4 seeds paired = 8 games
    qr.PRESETS["ablation_expander_smoke"] = replace(
        qr.PRESETS["qualification_smoke"],
        name="ablation_expander_smoke",
        seeds=8,
        opponents=("official_expander",),
    )
    qr.PRESETS["ablation_hunter_micro"] = replace(
        qr.PRESETS["qualification_hunter_info"],
        name="ablation_hunter_micro",
        seeds=4,
        opponents=("official_hunter",),
    )

    t0 = time.perf_counter()
    results = {"schema_version": 1, "kind": "PHASE_9Q_ABLATION_MATRIX", "candidates": {}}
    for name in CANDIDATES:
        print(f"=== {name} ===", flush=True)
        ch = _config_hash(name)
        exp = run_qualification_suite(
            policies=[name],
            preset_name="ablation_expander_smoke",
            out_path=REPO_ROOT / "experiments" / "manifests" / f"ablation_expander_{name}.json",
            wall_clock_s=wall_clock_s,
        )
        hun = run_qualification_suite(
            policies=[name],
            preset_name="ablation_hunter_micro",
            out_path=REPO_ROOT / "experiments" / "manifests" / f"ablation_hunter_{name}.json",
            wall_clock_s=wall_clock_s,
        )
        es = exp["policies"][name]["summary"]
        hs = hun["policies"][name]["summary"]
        row = {
            "config_hash": ch,
            "expander": {
                "W": es["wins"],
                "D": es["draws"],
                "L": es["losses"],
                "win_rate": es["win_rate"],
                "discovery_rate": es["enemy_general_discovery_rate"],
                "post_discovery_win_rate": es["post_discovery_win_rate"],
                "failure_classes": es["failure_classes"],
                "protocol_faults": sum(g.get("extras", {}).get("protocol_faults", 0) for g in exp.get("games", [])),
            },
            "hunter": {
                "W": hs["wins"],
                "D": hs["draws"],
                "L": hs["losses"],
                "loss_rate": hs["loss_rate"],
                "failure_classes": hs["failure_classes"],
            },
        }
        # Smoke gate
        exp_pass = es["losses"] == 0 and es["wins"] >= 12
        hun_pass = not (hs["wins"] == 0 and hs["losses"] == 8)
        row["smoke_expander_gate"] = "PASS" if exp_pass else "FAIL"
        row["smoke_hunter_gate"] = "PASS" if hun_pass else "FAIL"
        row["progress"] = bool(exp_pass and hun_pass)
        results["candidates"][name] = row
        print(
            f"  Expander {es['wins']}/{es['draws']}/{es['losses']} "
            f"Hunter {hs['wins']}/{hs['draws']}/{hs['losses']} "
            f"gates={row['smoke_expander_gate']}/{row['smoke_hunter_gate']}",
            flush=True,
        )
        if wall_clock_s is not None and (time.perf_counter() - t0) > wall_clock_s:
            results["stopped_early"] = True
            break

    results["elapsed_s"] = time.perf_counter() - t0
    # Best by expander wins then hunter wins
    ranked = sorted(
        results["candidates"].items(),
        key=lambda kv: (kv[1]["expander"]["W"], kv[1]["hunter"]["W"], -kv[1]["expander"]["D"]),
        reverse=True,
    )
    results["ranked"] = [n for n, _ in ranked]
    results["best"] = ranked[0][0] if ranked else None
    results["any_progress"] = any(v["progress"] for v in results["candidates"].values())
    out = REPO_ROOT / "experiments" / "manifests" / "phase_9q_ablation_matrix.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)
    return results


if __name__ == "__main__":
    # No wall-clock abort: full A–I Expander+Hunter smoke must finish.
    run_matrix(wall_clock_s=None)
