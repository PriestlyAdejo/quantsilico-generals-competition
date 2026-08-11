"""Rolling COMPLETE EMA Hunter canaries (newest-first, supersede backlog)."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CKPT = Path.home() / "quantsilico-runtime/emergency_rolling_v1/training/checkpoints"


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


def _list_complete_ema_candidates() -> list[Path]:
    out = []
    if not RUNTIME_CKPT.exists():
        return out
    for d in sorted(RUNTIME_CKPT.glob("ckpt_*")):
        if not (d / "COMPLETE").exists():
            continue
        if not (d / "ema.npz").exists():
            continue
        # skip tmp
        if d.name.endswith(".tmp"):
            continue
        out.append(d)
    # prefer higher update numbers
    def key(p: Path):
        meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        return int(meta.get("update", 0))

    return sorted(out, key=key)


def main() -> int:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_competition_native_jax_daytime_eval import (
        CNJPolicyAdapter,
        load_cnj_from_ckpt,
        play_game,
        OPPONENT_ALIASES,
    )
    from generals_bot.selector import create_policy

    report_path = ROOT / "experiments/manifests/emergency_rolling_ckpt_canary.json"
    if report_path.exists():
        doc = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        doc = {
            "schema_version": 1,
            "kind": "EMERGENCY_ROLLING_CKPT_CANARY",
            "evaluated": [],
            "pending": [],
            "superseded": [],
            "updated_at": None,
        }

    evaluated_ids = {e.get("checkpoint") for e in doc.get("evaluated", [])}
    cands = _list_complete_ema_candidates()
    # newest unevaluated
    target = None
    for c in reversed(cands):
        cid = str(c)
        if cid not in evaluated_ids and c.name not in {"ckpt_final"}:
            # prefer u768+ emergency
            target = c
            break
    if target is None:
        print("NO_UNEVALUATED_COMPLETE")
        _atomic_write_json(report_path, doc)
        return 0

    # supersede older pending
    for p in list(doc.get("pending") or []):
        doc.setdefault("superseded", []).append({**p, "status": "SUPERSEDED_UNEVALUATED"})
    doc["pending"] = [{"checkpoint": str(target), "which": "ema", "queued_at": datetime.now(timezone.utc).isoformat()}]

    meta = json.loads((target / "meta.json").read_text(encoding="utf-8"))
    policy = load_cnj_from_ckpt(target, which="ema")
    results = []
    t0 = time.perf_counter()
    for seed in (1201,):
        for seat in (0, 1):
            opp = create_policy(OPPONENT_ALIASES["official_hunter"], seed=seed)
            cand = CNJPolicyAdapter(policy.inner.weights, seed=seed + seat)
            p0, p1 = (cand, opp) if seat == 0 else (opp, cand)
            tg = time.perf_counter()
            out = play_game(p0, p1, seed=seed, max_turns=1200, focal_seat=seat)
            out.update(
                {
                    "key": f"roll:{target.name}:hunter:{seed}:focal{seat}",
                    "opponent": "official_hunter",
                    "seed": seed,
                    "focal_seat": seat,
                    "elapsed_s": time.perf_counter() - tg,
                    "checkpoint": str(target),
                    "which": "ema",
                    "update": meta.get("update"),
                }
            )
            results.append(out)
            print(out["key"], out.get("wdl"), flush=True)

    wins = sum(1 for r in results if r.get("wdl") == "win")
    draws = sum(1 for r in results if r.get("wdl") == "draw")
    losses = sum(1 for r in results if r.get("wdl") == "loss")
    triggers = []
    if draws:
        triggers.append("FIRST_HUNTER_DRAW")
    if wins:
        triggers.append("FIRST_HUNTER_WIN")

    entry = {
        "checkpoint": str(target),
        "name": target.name,
        "update": meta.get("update"),
        "which": "ema",
        "W": wins,
        "D": draws,
        "L": losses,
        "triggers": triggers,
        "results": results,
        "elapsed_s": time.perf_counter() - t0,
        "ts": datetime.now(timezone.utc).isoformat(),
        "baseline_note": "compare vs update-420 EMA hunter losses (prior canary score -18)",
    }
    # Stage 2 tiny tie-break if improved
    if wins or draws:
        seed = 1301
        seat = 0
        opp = create_policy(OPPONENT_ALIASES["official_expander"], seed=seed)
        cand = CNJPolicyAdapter(policy.inner.weights, seed=seed)
        out = play_game(cand, opp, seed=seed, max_turns=1200, focal_seat=seat)
        entry["stage2_expander"] = {
            "wdl": out.get("wdl"),
            "turns": out.get("turns"),
            "seed": seed,
            "seat": seat,
        }
        print("STAGE2 expander", out.get("wdl"), flush=True)

    doc.setdefault("evaluated", []).append(entry)
    doc["pending"] = []
    doc["status"] = "RUNNING"
    doc["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(report_path, doc)
    print(json.dumps({"evaluated": target.name, "W": wins, "D": draws, "L": losses, "triggers": triggers}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
