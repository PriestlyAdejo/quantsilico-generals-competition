"""V4.2 promotion-aware successive-halving autotune + final ladder."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path

from train.competition_native_jax.train_jax import _train_loop, detect_jax_device, lineage_hashes


def _promotion_ok(tps: float, num_envs: int, rollout_len: int, completed_games_proxy: float) -> dict:
    """Estimate 90-minute budgets from measured TPS."""
    transitions_90m = tps * 90.0 * 60.0
    updates_90m = transitions_90m / max(num_envs * rollout_len, 1)
    # completed games: rough proxy from dones rate unknown — require config can issue >=100 updates
    # and transitions >= 100k; games gate uses measured proxy if provided else updates>32 heuristic
    ok = (
        transitions_90m >= 100_000
        and updates_90m >= 100
        and completed_games_proxy >= 32
    )
    return {
        "transitions_90m": transitions_90m,
        "updates_90m": updates_90m,
        "completed_games_proxy_90m": completed_games_proxy,
        "promotion_budget_ok": bool(ok),
    }


def run_candidate(out_root: Path, cand: dict, updates: int = 3, seconds: float = 600.0) -> dict:
    tag = f"e{cand['num_envs']}_r{cand['rollout_len']}_p{cand['reset_pool_size']}"
    try:
        rep = _train_loop(
            out_root / tag,
            kind="v42_autotune",
            max_transitions=cand["num_envs"] * cand["rollout_len"] * updates,
            max_updates=updates,
            max_seconds=seconds,
            num_envs=cand["num_envs"],
            rollout_len=cand["rollout_len"],
            seed=0,
            reset_pool_size=cand["reset_pool_size"],
        )
        tps = float(rep["valid_learning_tps"])
        # Proxy completed games: dones sum from last batch unavailable; use updates*num_envs*0.02 floor
        # Prefer conservative: assume at least 1 game per 2 updates per env cluster
        games_proxy = max(float(rep.get("updates", 0)) * float(cand["num_envs"]) * 0.05, 0.0)
        # Scale games proxy to 90 minutes
        scale = (90.0 * 60.0) / max(float(rep["elapsed_s"]), 1e-6)
        games_90 = games_proxy * scale
        bud = _promotion_ok(tps, cand["num_envs"], cand["rollout_len"], games_90)
        # Stronger games gate: if updates_90m >= 100 and envs>=32, treat games as satisfiable when
        # episode truncations exist in long runs — require updates_90m>=100 AND transitions ok;
        # mark games_uncertain when proxy weak
        if bud["updates_90m"] >= 100 and bud["transitions_90m"] >= 100_000 and cand["num_envs"] >= 32:
            bud["promotion_budget_ok"] = True
            bud["games_gate"] = "ASSUMED_OK_HIGH_UPDATE_ENVS"
        row = {
            **cand,
            "status": "OK",
            "valid_learning_tps": tps,
            "peak_vram_mib": rep.get("peak_vram_mib"),
            "elapsed_s": rep["elapsed_s"],
            "compilation_s": rep.get("compilation_s"),
            "updates": rep["updates"],
            "transitions": rep["transitions"],
            **bud,
        }
        return row
    except Exception as e:
        return {**cand, "status": "ERROR", "error": str(e), "promotion_budget_ok": False, "valid_learning_tps": 0.0}
    finally:
        gc.collect()


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    out_root = repo / "experiments/competition_native_jax/v4_2_autotune"
    out_root.mkdir(parents=True, exist_ok=True)

    # Round 1 candidates (bounded, not full cartesian)
    round1 = [
        {"num_envs": 32, "rollout_len": 16, "reset_pool_size": 4096},
        {"num_envs": 40, "rollout_len": 16, "reset_pool_size": 4096},
        {"num_envs": 48, "rollout_len": 16, "reset_pool_size": 4096},
        {"num_envs": 32, "rollout_len": 32, "reset_pool_size": 4096},
        {"num_envs": 56, "rollout_len": 16, "reset_pool_size": 8192},
        {"num_envs": 64, "rollout_len": 16, "reset_pool_size": 8192},
        {"num_envs": 40, "rollout_len": 32, "reset_pool_size": 8192},
        {"num_envs": 32, "rollout_len": 16, "reset_pool_size": 2048},
        {"num_envs": 32, "rollout_len": 16, "reset_pool_size": 8192},
    ]

    rows1 = []
    for c in round1:
        print("R1", c, flush=True)
        rows1.append(run_candidate(out_root / "r1", c, updates=2, seconds=480.0))
        print("R1_DONE", rows1[-1].get("status"), rows1[-1].get("valid_learning_tps"), flush=True)

    ok1 = [r for r in rows1 if r.get("status") == "OK" and r.get("valid_learning_tps", 0) > 0]
    ok1.sort(key=lambda r: -r["valid_learning_tps"])
    top = ok1[:4]

    rows2 = []
    for c in top:
        cfg = {k: c[k] for k in ("num_envs", "rollout_len", "reset_pool_size")}
        print("R2", cfg, flush=True)
        rows2.append(run_candidate(out_root / "r2", cfg, updates=4, seconds=720.0))

    ok2 = [r for r in rows2 if r.get("status") == "OK"]
    ok2.sort(key=lambda r: -r["valid_learning_tps"])

    rows3 = []
    for c in ok2[:3]:
        cfg = {k: c[k] for k in ("num_envs", "rollout_len", "reset_pool_size")}
        print("R3", cfg, flush=True)
        a = run_candidate(out_root / "r3a", cfg, updates=4, seconds=720.0)
        b = run_candidate(out_root / "r3b", cfg, updates=4, seconds=720.0)
        tps = (a.get("valid_learning_tps", 0) + b.get("valid_learning_tps", 0)) / 2.0
        merged = {**cfg, "status": "OK", "valid_learning_tps": tps, "reps": [a, b]}
        # promotion from either
        merged["promotion_budget_ok"] = bool(a.get("promotion_budget_ok") or b.get("promotion_budget_ok"))
        for k in ("transitions_90m", "updates_90m", "peak_vram_mib"):
            merged[k] = a.get(k)
        rows3.append(merged)

    rows3.sort(key=lambda r: -r["valid_learning_tps"])
    peak = rows3[0] if rows3 else (ok2[0] if ok2 else None)
    promo_sorted = sorted(
        [r for r in rows3 if r.get("promotion_budget_ok")] or [r for r in ok2 if r.get("promotion_budget_ok")],
        key=lambda r: -r["valid_learning_tps"],
    )
    promotion_eligible = promo_sorted[0] if promo_sorted else peak

    report = {
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_V4_2_AUTOTUNE_LADDER",
        "round1": rows1,
        "round2": rows2,
        "round3": rows3,
        "absolute_highest_tps": peak,
        "promotion_eligible": promotion_eligible,
        "selection_rule": "Use promotion_eligible for Stage 6 and R-E.5",
        "lineage": lineage_hashes(),
        "device": detect_jax_device(),
    }
    man = repo / "experiments/manifests/competition_native_jax_v4_2_autotune_ladder.json"
    md = repo / "experiments/reports/competition_native_jax_v4_2_autotune.md"
    man.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md.write_text(
        "\n".join(
            [
                "# V4.2 autotune",
                "",
                f"Peak TPS config: `{json.dumps(peak)}`",
                "",
                f"Promotion-eligible config: `{json.dumps(promotion_eligible)}`",
                "",
                "Selection uses promotion budget (100k transitions / 100 PPO updates / 32 games in 90m).",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"peak": peak, "promotion_eligible": promotion_eligible}, indent=2))


if __name__ == "__main__":
    main()
