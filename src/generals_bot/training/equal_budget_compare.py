"""Bounded equal-budget DEVELOPMENT comparison: CNN control vs pure-torch graph.

Stops before INITIAL / OVERNIGHT / MARATHON. Never touches promotion holdout.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from generals_bot.training.ppo import run_bounded_ppo

REPO = Path(__file__).resolve().parents[3]


def run_equal_budget_dev_comparison(
    *,
    env_steps: int = 128,
    updates: int = 2,
    seed: int = 7,
    device: str | None = None,
    include_bc_init: bool = True,
) -> dict[str, Any]:
    """Run a tiny equal-budget PPO smoke comparison (DEVELOPMENT only)."""
    t0 = time.perf_counter()
    arms: list[dict[str, Any]] = [
        {"id": "cnn_scratch", "architecture": "recurrent_cnn_v2", "init": None},
        {
            "id": "graph_scratch",
            "architecture": "recurrent_graph_belief_v2_pure_torch",
            "init": None,
        },
    ]
    if include_bc_init:
        cnn_bc = REPO / "experiments" / "checkpoints" / "bc" / "tiny_recurrent_cnn_v2" / "model.json"
        graph_bc = (
            REPO / "experiments" / "checkpoints" / "bc" / "tiny_recurrent_graph_belief_v2" / "model.json"
        )
        if cnn_bc.is_file():
            arms.append(
                {
                    "id": "cnn_bc_init",
                    "architecture": "recurrent_cnn_v2",
                    "init": str(cnn_bc),
                }
            )
        if graph_bc.is_file():
            arms.append(
                {
                    "id": "graph_bc_init",
                    "architecture": "recurrent_graph_belief_v2_pure_torch",
                    "init": str(graph_bc),
                }
            )

    results: dict[str, Any] = {}
    for arm in arms:
        report = run_bounded_ppo(
            architecture=arm["architecture"],
            rollout_steps=env_steps,
            updates=updates,
            seed=seed,
            device=device,
            init_checkpoint=Path(arm["init"]) if arm["init"] else None,
            out_dir=REPO / "experiments" / "checkpoints" / "dev_compare" / arm["id"],
        )
        results[arm["id"]] = {
            "architecture": arm["architecture"],
            "init_checkpoint": arm["init"],
            "legal_action_rate": report.get("legal_action_rate"),
            "resume_ok": report.get("resume_ok"),
            "nan_free": report.get("nan_free"),
            "elapsed_s": report.get("elapsed_s"),
            "history": report.get("history"),
            "checkpoint": report.get("checkpoint"),
            "env_steps": env_steps * updates,
            "seed": seed,
            "split": "DEVELOPMENT",
            "note": "Equal-budget smoke only; not promotion evidence.",
        }

    summary = {
        "schema_version": 1,
        "kind": "EQUAL_BUDGET_DEVELOPMENT_COMPARISON",
        "budget": {
            "rollout_steps_per_update": env_steps,
            "updates": updates,
            "env_steps_per_arm": env_steps * updates,
            "seed": seed,
        },
        "arms": results,
        "pyg_needed": False,
        "elapsed_s": time.perf_counter() - t0,
        "stop_before": ["INITIAL", "OVERNIGHT", "MARATHON"],
        "notes": [
            "CNN is the learned control; graph-belief is the principal challenger.",
            "Training return alone is not evidence of improvement.",
            "No promotion holdout seeds used.",
        ],
    }
    out = REPO / "experiments" / "manifests" / "equal_budget_dev_comparison.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["path"] = str(out)
    return summary


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--env-steps", type=int, default=128)
    p.add_argument("--updates", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-bc-init", action="store_true")
    args = p.parse_args()
    summary = run_equal_budget_dev_comparison(
        env_steps=args.env_steps,
        updates=args.updates,
        seed=args.seed,
        include_bc_init=not args.no_bc_init,
    )
    print(json.dumps({k: summary[k] for k in summary if k != "arms"}, indent=2))
    for arm_id, arm in summary["arms"].items():
        print(arm_id, "legal", arm["legal_action_rate"], "elapsed", arm["elapsed_s"])


if __name__ == "__main__":
    main()
