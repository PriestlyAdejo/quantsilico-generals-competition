"""Bounded emergency canary: reuse frozen screening; ≤6 new Hunter-heavy games."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# Force CPU before jax import side effects in eval helpers
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    with tmp.open("rb") as f:
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-games", type=int, default=6)
    ap.add_argument("--wall-seconds", type=int, default=1800)
    ap.add_argument(
        "--ckpt",
        default="experiments/competition_native_jax/v4_3_r_e6/ckpt_final",
    )
    ap.add_argument("--which", default="raw", choices=("raw", "ema"))
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_competition_native_jax_daytime_eval import (  # type: ignore
        CNJPolicyAdapter,
        load_cnj_from_ckpt,
        play_game,
        OPPONENT_ALIASES,
    )
    from generals_bot.selector import create_policy

    frozen_path = ROOT / "experiments/manifests/competition_native_jax_r_e6_frozen_screening_partial.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.exists() else {}
    prior = []
    src = frozen.get("results") or {}
    if isinstance(src, dict):
        prior = list(src.get("results") or [])
    elif isinstance(src, list):
        prior = src

    reused = []
    for r in prior:
        if r.get("key"):
            reused.append(
                {
                    **r,
                    "existing_result_reused": True,
                    "source": "R_E6_FROZEN_SCREENING_PARTIAL",
                    "candidate": f"ckpt_final_ema",
                }
            )

    # Prefer Hunter games for new canary (raw final + seats)
    plan = []
    for seed in (1101, 1102, 1103):
        for seat in (0, 1):
            plan.append(("official_hunter", seed, seat))
    plan = plan[: args.max_new_games]

    ckpt = ROOT / args.ckpt
    # R-E.6 has no COMPLETE; triage allows legacy. New emergency ckpts require COMPLETE.
    if (ckpt / "COMPLETE").exists() or (ckpt / "meta.json").exists():
        pass
    else:
        raise SystemExit("CHECKPOINT_NOT_CONSUMABLE")

    t0 = time.perf_counter()
    policy = load_cnj_from_ckpt(ckpt, which=args.which)
    new_results = []
    for opp, seed, seat in plan:
        if time.perf_counter() - t0 > args.wall_seconds:
            break
        alias = OPPONENT_ALIASES.get(opp, opp)
        t_game = time.perf_counter()
        try:
            opp_pol = create_policy(alias, seed=seed)
            cand = CNJPolicyAdapter(policy.inner.weights, seed=seed + seat)
            if seat == 0:
                p0, p1 = cand, opp_pol
            else:
                p0, p1 = opp_pol, cand
            out = play_game(p0, p1, seed=seed, max_turns=1200, focal_seat=seat)
            out["elapsed_s"] = time.perf_counter() - t_game
            out["key"] = f"canary:{opp}:{seed}:focal{seat}"
            out["opponent"] = opp
            out["seed"] = seed
            out["focal_seat"] = seat
            out["candidate"] = f"{Path(args.ckpt).name}_{args.which}"
            out["existing_result_reused"] = False
            new_results.append(out)
            print("GAME", out["key"], out.get("wdl"), flush=True)
        except Exception as e:
            new_results.append(
                {
                    "key": f"canary:{opp}:{seed}:focal{seat}",
                    "error": str(e),
                    "wdl": "fault",
                    "existing_result_reused": False,
                }
            )

    # Rank: wins > draws > losses; Hunter wins matter most
    def score(rows):
        s = 0.0
        for r in rows:
            w = r.get("wdl")
            wgt = 3.0 if "hunter" in str(r.get("opponent", "")).lower() else 1.0
            if w == "win":
                s += 2 * wgt
            elif w == "draw":
                s += 1 * wgt
            elif w == "loss":
                s -= 1 * wgt
        return s

    cand_score = score(new_results)
    report = {
        "schema_version": 1,
        "kind": "EMERGENCY_BOOTSTRAP_CANARY",
        "status": "COMPLETE",
        "wall_seconds": args.wall_seconds,
        "elapsed_s": time.perf_counter() - t0,
        "max_new_games": args.max_new_games,
        "new_games_played": len(new_results),
        "reused_count": len(reused),
        "checkpoint": args.ckpt,
        "which": args.which,
        "canary_score": cand_score,
        "hunter_wins": sum(
            1
            for r in new_results
            if r.get("wdl") == "win" and "hunter" in str(r.get("opponent", "")).lower()
        ),
        "reused_results": reused,
        "new_results": new_results,
        "teacher_recommendation": "EMERGENCY_BOOTSTRAP_TEACHER" if new_results else "INSUFFICIENT",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    # Teacher: ckpt_final raw if any hunter activity; else ema from screening
    if report["hunter_wins"] > 0 or any(r.get("wdl") == "draw" for r in new_results):
        report["selected_teacher"] = {
            "checkpoint": args.ckpt,
            "which": args.which,
            "status": "EMERGENCY_BOOTSTRAP_TEACHER",
        }
    else:
        report["selected_teacher"] = {
            "checkpoint": "experiments/competition_native_jax/v4_3_r_e6/ckpt_final",
            "which": "ema",
            "status": "EMERGENCY_BOOTSTRAP_TEACHER",
            "note": "fallback to screened EMA (no canary improvement)",
        }

    out = ROOT / "experiments/manifests/emergency_bootstrap_canary.json"
    _atomic_write_json(out, report)
    print(json.dumps({"status": "COMPLETE", "score": cand_score, "new": len(new_results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
