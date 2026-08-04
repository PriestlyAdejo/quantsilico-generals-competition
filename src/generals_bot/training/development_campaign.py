"""Bounded DEVELOPMENT PPO with machine-readable hard limits.

Not INITIAL / OVERNIGHT / MARATHON. Holdout untouched.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from generals_bot.training.ppo import run_bounded_ppo
from generals_bot.training.telemetry_schema import annotate_history

REPO = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DevelopmentHardLimits:
    schema_version: int = 1
    kind: str = "DEVELOPMENT_CAMPAIGN_HARD_LIMITS"
    max_wall_clock_s_per_variant: float = 600.0
    max_total_wall_clock_s: float = 2400.0
    max_env_steps_per_variant: int = 512
    max_updates_per_variant: int = 4
    max_retained_checkpoints: int = 8
    disk_guardrail_mb: int = 2048
    thermal_stop: bool = True
    cancellation_supported: bool = False
    resume_supported: bool = True
    campaign: str = "DEVELOPMENT"
    forbidden: tuple[str, ...] = ("INITIAL", "OVERNIGHT", "MARATHON")


def run_development_compare(
    *,
    latency_classification: dict[str, str] | None = None,
    limits: DevelopmentHardLimits | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    limits = limits or DevelopmentHardLimits()
    seeds = seeds or [7, 11]
    latency_classification = latency_classification or {}
    t_global = time.perf_counter()

    arms: list[dict[str, Any]] = [
        {"id": "cnn_scratch", "architecture": "recurrent_cnn_v2", "init": None},
    ]
    cnn_bc = REPO / "experiments" / "checkpoints" / "bc" / "tiny_recurrent_cnn_v2" / "model.json"
    if cnn_bc.is_file():
        arms.append({"id": "cnn_bc_init", "architecture": "recurrent_cnn_v2", "init": str(cnn_bc)})

    graph_class = latency_classification.get("recurrent_graph_belief_v2", "UNKNOWN")
    graph_allowed = graph_class in {"PASS", "PARTIAL"}
    if graph_allowed:
        arms.append(
            {
                "id": "graph_scratch",
                "architecture": "recurrent_graph_belief_v2",
                "init": None,
            }
        )
        g_bc = REPO / "experiments" / "checkpoints" / "bc" / "tiny_recurrent_graph_belief_v2" / "model.json"
        if g_bc.is_file():
            arms.append(
                {
                    "id": "graph_bc_init",
                    "architecture": "recurrent_graph_belief_v2",
                    "init": str(g_bc),
                }
            )

    results: dict[str, Any] = {}
    stopped_reason = None
    for seed in seeds:
        if time.perf_counter() - t_global > limits.max_total_wall_clock_s:
            stopped_reason = "max_total_wall_clock"
            break
        for arm in arms:
            if time.perf_counter() - t_global > limits.max_total_wall_clock_s:
                stopped_reason = "max_total_wall_clock"
                break
            key = f"{arm['id']}_seed{seed}"
            t0 = time.perf_counter()
            try:
                report = run_bounded_ppo(
                    architecture=arm["architecture"],
                    rollout_steps=min(128, limits.max_env_steps_per_variant // max(limits.max_updates_per_variant, 1)),
                    updates=limits.max_updates_per_variant,
                    seed=seed,
                    device="cpu",
                    init_checkpoint=Path(arm["init"]) if arm["init"] else None,
                    out_dir=REPO / "experiments" / "checkpoints" / "dev_ppo" / key,
                )
                elapsed = time.perf_counter() - t0
                if elapsed > limits.max_wall_clock_s_per_variant:
                    stopped_reason = f"max_wall_clock_per_variant:{key}"
                results[key] = {
                    **{
                        k: report.get(k)
                        for k in (
                            "architecture",
                            "legal_action_rate",
                            "resume_ok",
                            "nan_free",
                            "history",
                            "checkpoint",
                        )
                    },
                    "telemetry": report.get("telemetry")
                    or annotate_history(report.get("history"), producer=f"development:{key}"),
                    "elapsed_s": elapsed,
                    "seed": seed,
                    "init": arm["init"],
                    "campaign": "DEVELOPMENT",
                    "env_steps": report.get("rollout_steps", 0) * report.get("updates", 0),
                }
                if report.get("legal_action_rate", 1.0) < 1.0 or not report.get("nan_free", True):
                    stopped_reason = f"quality_stop:{key}"
                    break
            except Exception as exc:  # noqa: BLE001
                results[key] = {"error": f"{type(exc).__name__}: {exc}", "campaign": "DEVELOPMENT"}
                stopped_reason = f"exception:{key}"
                break
        if stopped_reason:
            break

    summary = {
        "schema_version": 1,
        "kind": "BOUNDED_DEVELOPMENT_PPO",
        "limits": asdict(limits),
        "latency_classification": latency_classification,
        "graph_training_allowed": graph_allowed,
        "graph_deployment_status": (
            "ELIGIBLE_FOR_DEV"
            if graph_class == "PASS"
            else ("BLOCKED_FOR_DEPLOYMENT" if graph_class in {"PARTIAL", "FAIL"} else "UNKNOWN")
        ),
        "seeds": seeds,
        "exploratory": len(seeds) < 2,
        "arms": results,
        "stopped_reason": stopped_reason,
        "elapsed_s": time.perf_counter() - t_global,
        "holdout_used": False,
        "notes": [
            "Training return alone is not evidence of improvement.",
            "Not INITIAL/OVERNIGHT/MARATHON.",
        ],
    }
    out = REPO / "experiments" / "manifests" / "bounded_development_ppo.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["path"] = str(out)
    return summary


def main() -> None:
    lat_path = REPO / "experiments" / "manifests" / "competition_size_latency_gate.json"
    lat = {}
    if lat_path.is_file():
        lat = json.loads(lat_path.read_text(encoding="utf-8")).get("classification", {})
    summary = run_development_compare(latency_classification=lat)
    print(json.dumps({k: summary[k] for k in summary if k != "arms"}, indent=2))
    for k, v in summary["arms"].items():
        print(k, "legal", v.get("legal_action_rate"), "err", v.get("error"))


if __name__ == "__main__":
    main()
