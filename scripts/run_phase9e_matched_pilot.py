"""Phase 9E matched BASELINE vs CURRICULUM_ONLY pilot (sequential GPU arms)."""

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
from generals_bot.training.conversion_reward import RewardConfig
from generals_bot.training.device_policy import resolve_training_device
from generals_bot.training.ppo import run_bounded_ppo

REPO = Path(__file__).resolve().parents[1]
SEEDS = REPO / "experiments/manifests/seeds/phase9e_seed_partitions.json"
PILOT_CFG = REPO / "configs/training/phase9e/pilot_v1.yaml"
OUT = REPO / "experiments/manifests/phase9e_matched_pilot.json"
READINESS = REPO / "experiments/manifests/phase9e_readiness_gate.json"

CNN_CKPT = REPO / "experiments/checkpoints/initial/initial_cnn_cnn_bc_init_seed11/chunk_0/model.json"
GRAPH_CKPT = REPO / "experiments/checkpoints/initial/initial_graph_graph_bc_init_seed7/chunk_0/model.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_reward(name: str) -> RewardConfig:
    path = REPO / "configs/training/phase9e" / f"{name}.yaml"
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


def assert_step_zero_equality(a: dict[str, Any], b: dict[str, Any], arch: str) -> None:
    keys = ("wins", "draws", "losses", "protocol_faults", "score_rate")
    for k in keys:
        if a[k] != b[k]:
            raise RuntimeError(f"step-zero mismatch for {arch}: {k} {a[k]} != {b[k]}")


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
    out_root = REPO / "experiments/checkpoints/phase9e" / arm_id
    out_root.mkdir(parents=True, exist_ok=True)

    z0 = step_zero_eval(architecture, frozen_ckpt, monitor_seeds[:2], device)
    print(
        json.dumps(
            {
                "arm": arm_id,
                "step_zero": {k: z0[k] for k in ("wins", "draws", "losses", "protocol_faults", "score_rate")},
            }
        ),
        flush=True,
    )

    budgets = cfg["budgets"]
    target = int(budgets["target_env_steps"])
    max_steps = int(budgets["max_env_steps"])
    wall_cap = float(budgets["max_wall_clock_min_per_arm"]) * 60.0
    chunk_steps = int(budgets["chunk_env_steps"])
    updates = int(budgets["chunk_updates"])
    rollout = int(budgets["rollout_steps"])
    every = int(budgets["validation_interval_chunks"])

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
        env_steps += rollout * updates  # equals chunk_env_steps when matched
        assert rollout * updates == chunk_steps
        chunks += 1
        print(
            json.dumps(
                {
                    "arm": arm_id,
                    "chunks": chunks,
                    "env_steps": env_steps,
                    "device": report.get("device"),
                    "elapsed_s": round(time.perf_counter() - t0, 1),
                }
            ),
            flush=True,
        )
        if chunks % every == 0:
            val = step_zero_eval(architecture, ckpt, monitor_seeds, device)
            windows.append({"env_steps": env_steps, "chunk": chunks, "validation": val})
            print(
                json.dumps(
                    {
                        "arm": arm_id,
                        "validation": {
                            k: val[k] for k in ("wins", "draws", "losses", "protocol_faults", "score_rate")
                        },
                    }
                ),
                flush=True,
            )
            if val["protocol_faults"] > 0:
                stop_reason = "PROTOCOL_FAULT"
                break

    if not windows:
        val = step_zero_eval(architecture, ckpt, monitor_seeds, device)
        windows.append({"env_steps": env_steps, "chunk": chunks, "validation": val})

    best = max(windows, key=lambda w: (w["validation"]["wins"], w["validation"]["score_rate"]))
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


def classify_pair(baseline: dict[str, Any], curriculum: dict[str, Any], epsilon: float) -> str:
    b_best = (baseline.get("best_window") or {}).get("validation") or baseline["step_zero"]
    c_best = (curriculum.get("best_window") or {}).get("validation") or curriculum["step_zero"]
    b_wins, c_wins = int(b_best["wins"]), int(c_best["wins"])
    b_sr, c_sr = float(b_best["score_rate"]), float(c_best["score_rate"])
    if c_wins > b_wins and c_sr >= b_sr:
        return "INTERVENTION_EFFECT"
    if c_sr >= b_sr + epsilon and c_wins >= b_wins:
        return "INTERVENTION_EFFECT"
    if c_wins > 0 and b_wins > 0 and abs(c_sr - b_sr) < epsilon:
        return "BOTH_IMPROVED"
    if c_wins == 0 and b_wins > 0:
        return "REGRESSION"
    if c_wins == 0 and b_wins == 0 and abs(c_sr - b_sr) < epsilon:
        return "NO_MEANINGFUL_IMPROVEMENT"
    return "INSUFFICIENT_EVIDENCE"


def promising(curriculum: dict[str, Any], baseline: dict[str, Any], effect: str) -> bool:
    """Win in each of at least two distinct monitoring windows after separate intervals."""
    if effect != "INTERVENTION_EFFECT":
        return False
    windows = curriculum.get("validation_windows") or []
    win_windows = [w for w in windows if int(w["validation"]["wins"]) >= 1 and int(w["validation"]["protocol_faults"]) == 0]
    if len(win_windows) < 2:
        return False
    # distinct env_steps intervals
    steps = {int(w["env_steps"]) for w in win_windows}
    if len(steps) < 2:
        return False
    b_best = (baseline.get("best_window") or {}).get("validation") or baseline["step_zero"]
    c_best = (curriculum.get("best_window") or {}).get("validation")
    if not c_best:
        return False
    if int(c_best["wins"]) <= int(b_best["wins"]):
        return False
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["cnn", "graph", "both"], default="both")
    args = p.parse_args()

    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    if readiness.get("decision") != "PASS":
        raise SystemExit(f"PHASE9E_READINESS_GATE not PASS: {readiness.get('decision')}")

    cfg = yaml.safe_load(PILOT_CFG.read_text(encoding="utf-8"))
    seeds = json.loads(SEEDS.read_text(encoding="utf-8"))["partitions"]
    train_seed = int(seeds["full_game_training_seeds"]["seeds"][0])
    monitor = list(seeds["full_game_monitoring_seeds"]["seeds"])
    device = resolve_training_device(cfg["optimisation"].get("device", "cuda:0"), context="phase9e_pilot")

    arms_spec = []
    if args.arch in {"cnn", "both"}:
        arms_spec.extend(
            [
                ("cnn_baseline", "recurrent_cnn_v2", CNN_CKPT, "baseline_v1"),
                ("cnn_curriculum", "recurrent_cnn_v2", CNN_CKPT, "curriculum_discovery_v1"),
            ]
        )
    if args.arch in {"graph", "both"}:
        arms_spec.extend(
            [
                ("graph_baseline", "recurrent_graph_belief_v2", GRAPH_CKPT, "baseline_v1"),
                ("graph_curriculum", "recurrent_graph_belief_v2", GRAPH_CKPT, "curriculum_discovery_v1"),
            ]
        )

    # Step-zero equality per architecture before training
    for arch_name, arch_id, ckpt in (
        ("cnn", "recurrent_cnn_v2", CNN_CKPT),
        ("graph", "recurrent_graph_belief_v2", GRAPH_CKPT),
    ):
        if args.arch not in {arch_name, "both"}:
            continue
        a = step_zero_eval(arch_id, ckpt, monitor[:2], device)
        b = step_zero_eval(arch_id, ckpt, monitor[:2], device)
        assert_step_zero_equality(a, b, arch_name)
        print(json.dumps({"step_zero_equality": arch_name, "status": "PASS", "score_rate": a["score_rate"]}), flush=True)

    results: dict[str, Any] = {}
    for arm_id, architecture, ckpt, treatment in arms_spec:
        results[arm_id] = run_arm(
            arm_id=arm_id,
            architecture=architecture,
            frozen_ckpt=ckpt,
            treatment=treatment,
            continuation_seed=train_seed,
            monitor_seeds=monitor,
            cfg=cfg,
            device=device,
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "PHASE9E_MATCHED_PILOT",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "pilot_config": str(PILOT_CFG),
        "causal_design": "BASELINE_vs_CURRICULUM_ONLY",
        "teacher_refresh": False,
        "arms": results,
        "pairs": {},
    }

    if "cnn_baseline" in results and "cnn_curriculum" in results:
        effect = classify_pair(results["cnn_baseline"], results["cnn_curriculum"], 0.05)
        report["pairs"]["cnn"] = {
            "effect": effect,
            "promising": promising(results["cnn_curriculum"], results["cnn_baseline"], effect),
        }
    if "graph_baseline" in results and "graph_curriculum" in results:
        effect = classify_pair(results["graph_baseline"], results["graph_curriculum"], 0.05)
        report["pairs"]["graph"] = {
            "effect": effect,
            "promising": promising(results["graph_curriculum"], results["graph_baseline"], effect),
        }

    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "pairs": report["pairs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
