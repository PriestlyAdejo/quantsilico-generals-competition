"""Immutable DEVELOPMENT eight-arm audit → ranked candidates for INITIAL."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = REPO / "experiments" / "manifests" / "bounded_development_ppo.json"
DEFAULT_OUT = REPO / "experiments" / "manifests" / "development_arm_audit.json"


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _arm_metrics(arm_id: str, arm: dict[str, Any]) -> dict[str, Any]:
    history = arm.get("history") or []
    explained = [float(r["explained_variance"]) for r in history if r.get("explained_variance") is not None]
    losses = [float(r["loss"]) for r in history if r.get("loss") is not None]
    final_loss = losses[-1] if losses else None
    first_loss = losses[0] if losses else None
    loss_delta = (final_loss - first_loss) if final_loss is not None and first_loss is not None else None
    ckpt = arm.get("checkpoint")
    ckpt_ok = bool(ckpt and Path(str(ckpt)).is_file())
    legal = float(arm.get("legal_action_rate") or 0.0)
    resume_ok = bool(arm.get("resume_ok"))
    nan_free = bool(arm.get("nan_free"))
    eng_ok = legal >= 1.0 and resume_ok and nan_free and ckpt_ok
    # Validation score rate is the primary hierarchy key when present.
    val_score = arm.get("validation_score_rate")
    if val_score is None and isinstance(arm.get("validation"), dict):
        val_score = arm["validation"].get("score_rate")
    return {
        "arm_id": arm_id,
        "architecture": arm.get("architecture"),
        "seed": arm.get("seed"),
        "init": arm.get("init"),
        "legal_action_rate": legal,
        "resume_ok": resume_ok,
        "nan_free": nan_free,
        "checkpoint": ckpt,
        "checkpoint_exists": ckpt_ok,
        "engineering_ok": eng_ok,
        "env_steps": arm.get("env_steps"),
        "elapsed_s": arm.get("elapsed_s"),
        "validation_score_rate": val_score,
        "mean_explained_variance": _mean(explained),
        "final_loss": final_loss,
        "loss_delta": loss_delta,
        "updates": len(history),
    }


def _rank_key(row: dict[str, Any]) -> tuple:
    """Higher is better. Validation score rate dominates when recorded."""
    val = row.get("validation_score_rate")
    has_val = 1 if isinstance(val, (int, float)) else 0
    val_v = float(val) if has_val else 0.0
    eng = 1 if row.get("engineering_ok") else 0
    bc = 1 if row.get("init") else 0
    ev = float(row.get("mean_explained_variance") or -1e9)
    # Prefer smaller loss growth (negative delta better); invert for sort
    delta = row.get("loss_delta")
    inv_delta = -float(delta) if isinstance(delta, (int, float)) else -1e9
    return (eng, has_val, val_v, bc, ev, inv_delta)


def audit_development_arms(source: Path = DEFAULT_SOURCE) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(f"DEVELOPMENT manifest missing: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("kind") != "BOUNDED_DEVELOPMENT_PPO":
        raise ValueError(f"Unexpected kind: {payload.get('kind')}")
    arms = payload.get("arms") or {}
    rows = [_arm_metrics(aid, arm) for aid, arm in arms.items()]
    ranked = sorted(rows, key=_rank_key, reverse=True)

    cnn = [r for r in ranked if r.get("architecture") == "recurrent_cnn_v2" and r["engineering_ok"]]
    graph = [
        r for r in ranked if r.get("architecture") == "recurrent_graph_belief_v2" and r["engineering_ok"]
    ]
    best_cnn = cnn[0] if cnn else None
    best_graph = graph[0] if graph else None

    hierarchy_note = (
        "Rank order: engineering_ok → validation_score_rate (if present) → "
        "BC init preferred → mean explained_variance → lower loss growth. "
        "DEVELOPMENT arms in this manifest do not record validation_score_rate."
    )

    return {
        "schema_version": 1,
        "kind": "DEVELOPMENT_ARM_AUDIT",
        "source_manifest": str(source.relative_to(REPO)).replace("\\", "/"),
        "source_kind": payload.get("kind"),
        "immutable": True,
        "arm_count": len(rows),
        "ranked_arms": ranked,
        "best_cnn": best_cnn,
        "best_graph": best_graph,
        "graph_training_allowed": bool(payload.get("graph_training_allowed")),
        "latency_classification": payload.get("latency_classification"),
        "hierarchy_note": hierarchy_note,
        "audited_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    report = audit_development_arms(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": "RECORDED", "path": str(args.out), "best_cnn": (report.get("best_cnn") or {}).get("arm_id"), "best_graph": (report.get("best_graph") or {}).get("arm_id")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
