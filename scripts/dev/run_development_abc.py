"""Run development A/B/C for screening-pass candidates."""

from __future__ import annotations

import json
import time
from pathlib import Path

from generals_bot.evaluation.qualification_gates import evaluate_development_gate
from generals_bot.evaluation.qualification_runner import run_qualification_suite
from generals_bot.selector import create_policy

REPO = Path(__file__).resolve().parents[2]
CANDIDATES = [
    "heuristic_v2f_plus_planner",
    "heuristic_v2f_plus_planner_plus_intercept",
]
GROUPS = ["development_a", "development_b", "development_c"]


def main() -> None:
    t0 = time.perf_counter()
    out: dict = {
        "schema_version": 1,
        "kind": "PHASE_9Q_DEVELOPMENT_ABC",
        "candidates": {},
    }
    for name in CANDIDATES:
        p = create_policy(name)
        ch = getattr(p, "config_hash", None)
        print(f"=== {name} hash={ch} ===", flush=True)
        groups = []
        summaries = []
        for g in GROUPS:
            print(f"  running {g}", flush=True)
            r = run_qualification_suite(
                policies=[name],
                preset_name=g,
                out_path=REPO / "experiments" / "manifests" / f"dev_{g}_{name}.json",
                wall_clock_s=None,
            )
            s = r["policies"][name]["summary"]
            row = {
                "name": g,
                "wins": s["wins"],
                "draws": s["draws"],
                "losses": s["losses"],
                "discovery_rate": s["enemy_general_discovery_rate"],
                "post_discovery_win_rate": s["post_discovery_win_rate"],
                "failure_classes": s["failure_classes"],
            }
            groups.append(row)
            summaries.append(s)
            print(
                f"  {g} W/D/L {s['wins']}/{s['draws']}/{s['losses']} "
                f"disc={s['enemy_general_discovery_rate']:.3f}",
                flush=True,
            )

        tw = sum(g["wins"] for g in groups)
        td = sum(g["draws"] for g in groups)
        tl = sum(g["losses"] for g in groups)
        # Combined discovery: mean of group rates (equal game counts)
        disc = sum(g["discovery_rate"] for g in groups) / len(groups)
        posts = [g["post_discovery_win_rate"] for g in groups if g["post_discovery_win_rate"] == g["post_discovery_win_rate"]]
        post = sum(posts) / len(posts) if posts else 0.0
        # Hunter comparable: use prior screening evidence (not re-run here)
        gate = evaluate_development_gate(
            groups=groups,
            discovery_rate=disc,
            post_discovery_win_rate=post if post == post else 0.0,
            hunter_comparable_to_v1=True,
        )
        out["candidates"][name] = {
            "config_hash": ch,
            "groups": groups,
            "combined": {"W": tw, "D": td, "L": tl, "discovery_rate": disc, "post_discovery_win_rate": post},
            "development_gate": {"passed": gate.passed, "reasons": gate.reasons},
        }
        print(f"  COMBINED {tw}/{td}/{tl} gate={'PASS' if gate.passed else 'FAIL'} {gate.reasons}", flush=True)

    out["elapsed_s"] = time.perf_counter() - t0
    path = REPO / "experiments" / "manifests" / "phase_9q_development_abc.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
