"""Phase 9D matched four-arm draw-conversion pilot (sequential)."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from generals_bot.training.adaptive_initial import validate_checkpoint_vs_expander
from generals_bot.training.conversion_reward import CONVERSION_V1, CONTROL_V1, RewardConfig
from generals_bot.training.ppo import run_bounded_ppo

REPO = Path(__file__).resolve().parents[1]
SEEDS = REPO / "experiments/manifests/seeds/phase9d_seed_partitions.json"
PILOT_CFG = REPO / "configs/training/draw_conversion/pilot_v1.yaml"
OUT = REPO / "experiments/manifests/phase9d_matched_pilot.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_reward(name: str) -> RewardConfig:
    if name == "control":
        path = REPO / "configs/training/draw_conversion/control_v1.yaml"
    else:
        path = REPO / "configs/training/draw_conversion/conversion_v1.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RewardConfig.from_dict(data)


def step_zero_eval(architecture: str, checkpoint: Path, seeds: list[int], device: str) -> dict[str, Any]:
    seat0 = validate_checkpoint_vs_expander(
        architecture=architecture,
        checkpoint=checkpoint,
        seeds=seeds,
        max_turns=100,
        device=device,
        learned_seat=0,
    )
    seat1 = validate_checkpoint_vs_expander(
        architecture=architecture,
        checkpoint=checkpoint,
        seeds=seeds,
        max_turns=100,
        device=device,
        learned_seat=1,
    )
    return {
        "seat0": seat0,
        "seat1": seat1,
        "protocol_faults": seat0["protocol_faults"] + seat1["protocol_faults"],
        "wins": seat0["wins"] + seat1["wins"],
        "draws": seat0["draws"] + seat1["draws"],
        "losses": seat0["losses"] + seat1["losses"],
        "score_rate": (
            (seat0["wins"] + seat1["wins"] + 0.5 * (seat0["draws"] + seat1["draws"]))
            / max(seat0["games"] + seat1["games"], 1)
        ),
    }


def run_arm(
    *,
    arm_id: str,
    architecture: str,
    frozen_ckpt: Path,
    treatment: str,
    continuation_seed: int,
    monitor_seeds: list[int],
    cfg: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    reward = _load_reward(treatment)
    out_root = REPO / "experiments/checkpoints/draw_conversion" / arm_id
    out_root.mkdir(parents=True, exist_ok=True)

    # Step-zero baseline (same frozen ckpt for control/conversion of same arch)
    z0 = step_zero_eval(architecture, frozen_ckpt, monitor_seeds[:2], device)
    print(json.dumps({"arm": arm_id, "step_zero": {k: z0[k] for k in ("wins", "draws", "losses", "protocol_faults", "score_rate")}}), flush=True)

    budgets = cfg["budgets"]
    target = int(budgets["target_env_steps"])
    max_steps = int(budgets["max_env_steps"])
    wall_cap = float(budgets["wall_clock_min_per_arm"]) * 60.0
    chunk_steps = int(budgets["chunk_env_steps"])
    updates = int(budgets["chunk_updates"])
    rollout = int(budgets["rollout_steps"])
    every = int(budgets["validation_every_chunks"])

    ckpt = frozen_ckpt
    env_steps = 0
    chunks = 0
    windows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    stop_reason = "BUDGET"

    while env_steps < target and env_steps < max_steps:
        if time.perf_counter() - t0 > wall_cap:
            stop_reason = "WALL_CLOCK"
            break
        chunk_dir = out_root / f"chunk_{chunks}"
        report = run_bounded_ppo(
            architecture=architecture,
            rollout_steps=rollout,
            updates=updates,
            lr=float(cfg["optimisation"]["lr"]),
            clip=float(cfg["optimisation"]["clip"]),
            seed=continuation_seed + chunks,
            device=device,
            init_checkpoint=ckpt,
            out_dir=chunk_dir,
            reward_config=reward,
        )
        ckpt = Path(report["checkpoint"])
        env_steps += rollout * updates
        chunks += 1
        print(json.dumps({"arm": arm_id, "chunks": chunks, "env_steps": env_steps, "elapsed_s": round(time.perf_counter() - t0, 1)}), flush=True)

        if chunks % every == 0:
            val = step_zero_eval(architecture, ckpt, monitor_seeds, device)
            windows.append({"env_steps": env_steps, "chunk": chunks, "validation": val})
            print(json.dumps({"arm": arm_id, "validation": {k: val[k] for k in ("wins", "draws", "losses", "protocol_faults", "score_rate")}}), flush=True)
            if val["protocol_faults"] > 0:
                stop_reason = "PROTOCOL_FAULT"
                break

    if len(windows) < int(budgets["min_validation_windows"]) and stop_reason == "BUDGET":
        # Force a final validation window
        val = step_zero_eval(architecture, ckpt, monitor_seeds, device)
        windows.append({"env_steps": env_steps, "chunk": chunks, "validation": val})

    best = max(windows, key=lambda w: (w["validation"]["wins"], w["validation"]["score_rate"])) if windows else None
    return {
        "arm_id": arm_id,
        "architecture": architecture,
        "treatment": treatment,
        "reward_version": reward.version,
        "frozen_checkpoint": str(frozen_ckpt),
        "frozen_sha256": _sha(frozen_ckpt),
        "final_checkpoint": str(ckpt),
        "final_sha256": _sha(ckpt) if ckpt.is_file() else None,
        "continuation_seed": continuation_seed,
        "monitor_seeds": monitor_seeds,
        "step_zero": z0,
        "env_steps": env_steps,
        "chunks": chunks,
        "elapsed_s": time.perf_counter() - t0,
        "stop_reason": stop_reason,
        "validation_windows": windows,
        "best_window": best,
    }


def classify_pair(control: dict[str, Any], conversion: dict[str, Any], epsilon: float) -> str:
    c_best = (control.get("best_window") or {}).get("validation") or control["step_zero"]
    v_best = (conversion.get("best_window") or {}).get("validation") or conversion["step_zero"]
    c_wins, v_wins = int(c_best["wins"]), int(v_best["wins"])
    c_sr, v_sr = float(c_best["score_rate"]), float(v_best["score_rate"])
    if v_wins > 0 and c_wins == 0:
        return "INTERVENTION_EFFECT"
    if v_sr >= c_sr + epsilon and v_wins >= c_wins:
        return "INTERVENTION_EFFECT"
    if v_wins > 0 and c_wins > 0 and abs(v_sr - c_sr) < epsilon:
        return "BOTH_IMPROVED"
    if v_wins == 0 and c_wins > 0:
        return "REGRESSION"
    if v_wins == 0 and c_wins == 0 and abs(v_sr - c_sr) < epsilon:
        # Extra training may have helped score_rate slightly without wins
        if max(v_sr, c_sr) > float(control["step_zero"]["score_rate"]) + epsilon:
            return "TRAINING_BUDGET_EFFECT"
        return "NO_MEANINGFUL_IMPROVEMENT"
    if v_wins > 0 and c_wins > 0 and v_sr > c_sr:
        return "INTERVENTION_EFFECT"
    return "INSUFFICIENT_EVIDENCE"


def promising(arm: dict[str, Any], control: dict[str, Any], effect: str) -> bool:
    best = (arm.get("best_window") or {}).get("validation")
    if not best:
        return False
    if int(best["protocol_faults"]) != 0:
        return False
    if int(best["wins"]) < 1:
        return False
    win_windows = sum(1 for w in arm["validation_windows"] if int(w["validation"]["wins"]) > 0)
    if win_windows < 2:
        return False
    return effect in {"INTERVENTION_EFFECT", "BOTH_IMPROVED", "TRAINING_BUDGET_EFFECT"} and effect == "INTERVENTION_EFFECT"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None)
    p.add_argument("--arms", default="all", help="all|cnn_control|cnn_conversion|graph_control|graph_conversion")
    args = p.parse_args()
    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    cfg = yaml.safe_load(PILOT_CFG.read_text(encoding="utf-8"))
    seeds_doc = json.loads(SEEDS.read_text(encoding="utf-8"))
    cont = seeds_doc["partitions"]["continuation_training_seeds"]["seeds"][0]
    monitor = seeds_doc["partitions"]["pilot_monitoring_seeds"]["seeds"]

    arms_spec = [
        ("cnn_control", "recurrent_cnn_v2", REPO / "experiments/checkpoints/initial/initial_cnn_cnn_bc_init_seed11/chunk_0/model.json", "control"),
        ("cnn_conversion", "recurrent_cnn_v2", REPO / "experiments/checkpoints/initial/initial_cnn_cnn_bc_init_seed11/chunk_0/model.json", "conversion"),
        ("graph_control", "recurrent_graph_belief_v2", REPO / "experiments/checkpoints/initial/initial_graph_graph_bc_init_seed7/chunk_0/model.json", "control"),
        ("graph_conversion", "recurrent_graph_belief_v2", REPO / "experiments/checkpoints/initial/initial_graph_graph_bc_init_seed7/chunk_0/model.json", "conversion"),
    ]
    if args.arms != "all":
        arms_spec = [a for a in arms_spec if a[0] == args.arms]

    # Step-zero equality gate across control/conversion per architecture before training
    for arch_name, ckpt in (
        ("cnn", arms_spec[0][2] if arms_spec else None),
        ("graph", None),
    ):
        pass

    results: dict[str, Any] = {}
    # Full equality check when running all
    if args.arms == "all":
        cnn_c = step_zero_eval("recurrent_cnn_v2", arms_spec[0][2], monitor[:2], device)
        cnn_v = step_zero_eval("recurrent_cnn_v2", arms_spec[1][2], monitor[:2], device)
        graph_c = step_zero_eval("recurrent_graph_belief_v2", arms_spec[2][2], monitor[:2], device)
        graph_v = step_zero_eval("recurrent_graph_belief_v2", arms_spec[3][2], monitor[:2], device)
        eq = {
            "cnn_match": cnn_c["wins"] == cnn_v["wins"] and cnn_c["draws"] == cnn_v["draws"] and cnn_c["losses"] == cnn_v["losses"] and cnn_c["protocol_faults"] == cnn_v["protocol_faults"],
            "graph_match": graph_c["wins"] == graph_v["wins"] and graph_c["draws"] == graph_v["draws"] and graph_c["losses"] == graph_v["losses"] and graph_c["protocol_faults"] == graph_v["protocol_faults"],
            "cnn_control": cnn_c,
            "cnn_conversion": cnn_v,
            "graph_control": graph_c,
            "graph_conversion": graph_v,
        }
        print(json.dumps({"step_zero_equality": {"cnn_match": eq["cnn_match"], "graph_match": eq["graph_match"]}}), flush=True)
        if not eq["cnn_match"] or not eq["graph_match"]:
            raise SystemExit("Step-zero CONTROL/CONVERSION mismatch — matched comparison blocked")
        results["step_zero_equality"] = eq

    for arm_id, architecture, ckpt, treatment in arms_spec:
        print(f"ARM_START {arm_id}", flush=True)
        results[arm_id] = run_arm(
            arm_id=arm_id,
            architecture=architecture,
            frozen_ckpt=ckpt,
            treatment=treatment,
            continuation_seed=cont,
            monitor_seeds=monitor,
            cfg=cfg,
            device=device,
        )
        # Persist partial after each arm
        partial = {
            "schema_version": 1,
            "kind": "PHASE9D_MATCHED_PILOT",
            "status": "RUNNING",
            "arms": {k: v for k, v in results.items() if k.endswith(("control", "conversion")) or k == "step_zero_equality"},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        OUT.write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")

    if args.arms == "all" and "cnn_control" in results and "cnn_conversion" in results:
        eps = float(cfg["treatment_effect"]["score_rate_epsilon"])
        cnn_effect = classify_pair(results["cnn_control"], results["cnn_conversion"], eps)
        graph_effect = classify_pair(results["graph_control"], results["graph_conversion"], eps)
        results["treatment_effects"] = {
            "cnn": cnn_effect,
            "graph": graph_effect,
            "cnn_promising_for_matched_replication": promising(results["cnn_conversion"], results["cnn_control"], cnn_effect),
            "graph_promising_for_matched_replication": promising(results["graph_conversion"], results["graph_control"], graph_effect),
        }

    payload = {
        "schema_version": 1,
        "kind": "PHASE9D_MATCHED_PILOT",
        "gate_name": "PHASE9D_MATCHED_PILOT",
        "research_generation_id": "phase9d_draw_conversion_2026-08-04",
        "decision": "COMPLETE",
        "status": "COMPLETE",
        "continuation_seed": cont,
        "pilot_monitoring_seeds": monitor,
        "pilot_config": "configs/training/draw_conversion/pilot_v1.yaml",
        "arms": {k: v for k, v in results.items() if k.endswith(("control", "conversion"))},
        "step_zero_equality": results.get("step_zero_equality"),
        "treatment_effects": results.get("treatment_effects"),
        "engine_sha": "9e3b9d13cca51caa1bb07db48bb85c9e90ce0462",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "supersedes": None,
        "superseded_by": None,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("WROTE", OUT, flush=True)
    print(json.dumps(results.get("treatment_effects"), indent=2), flush=True)


if __name__ == "__main__":
    main()
