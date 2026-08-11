"""PPO learning-signal classifier (ops resume/telemetry; never stops for WEAK/STAGNATION)."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path.home() / "quantsilico-runtime" / "emergency_rolling_v1"
METRICS = RUNTIME / "training" / "metrics" / "latest.jsonl"


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


def _load_rows(limit: int = 500) -> list[dict]:
    if not METRICS.exists():
        # also try heartbeat json
        hb = RUNTIME / "training" / "metrics"
        rows = []
        if hb.exists():
            for p in sorted(hb.glob("*.json"))[-limit:]:
                try:
                    rows.append(json.loads(p.read_text(encoding="utf-8")))
                except Exception:
                    pass
        prog = ROOT / "experiments/manifests/emergency_rolling_programme_state.json"
        if prog.exists():
            rows.append(json.loads(prog.read_text(encoding="utf-8")))
        return rows[-limit:]
    rows = []
    with METRICS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]


def classify(window: list[dict]) -> str:
    if not window:
        return "WEAK_CONTINUE"
    # numerical failure: non-finite loss/entropy
    for r in window:
        for k in ("loss", "entropy", "pg", "vloss", "ratio"):
            v = r.get(k)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fv):
                return "NUMERICAL_FAILURE"
    # stagnation heuristic: entropy near-zero and loss flat
    ents = [float(r["entropy"]) for r in window if r.get("entropy") is not None]
    losses = [float(r["loss"]) for r in window if r.get("loss") is not None]
    if ents and max(ents) < 1e-4:
        return "POSSIBLE_STAGNATION"
    if losses and ents:
        # healthy if entropy in reasonable band and losses finite
        mean_e = sum(ents) / len(ents)
        if 0.05 <= mean_e <= 8.0:
            return "HEALTHY"
        return "WEAK_CONTINUE"
    return "WEAK_CONTINUE"


def main() -> int:
    rows = _load_rows()
    # classify every 128 updates when possible
    by_update = {}
    for r in rows:
        u = r.get("updates") or r.get("update")
        if u is not None:
            by_update[int(u)] = r
    updates = sorted(by_update)
    classifications = []
    for i, u in enumerate(updates):
        if u % 128 != 0 and i != len(updates) - 1:
            continue
        window = [by_update[x] for x in updates if u - 128 < x <= u]
        classifications.append(
            {
                "update": u,
                "class": classify(window),
                "n_rows": len(window),
            }
        )
    latest = classifications[-1]["class"] if classifications else classify(rows[-16:])
    doc = {
        "schema_version": 1,
        "kind": "EMERGENCY_PPO_LEARNING_SIGNAL",
        "latest_class": latest,
        "policy": "never_stop_for_WEAK_CONTINUE_or_POSSIBLE_STAGNATION; only NUMERICAL_FAILURE stops",
        "classifications": classifications[-32:],
        "n_metric_rows": len(rows),
        "note": "Attached on next cooperative resume; does not restart PID for telemetry alone.",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(ROOT / "experiments/manifests/emergency_ppo_learning_signal.json", doc)
    md = ROOT / "experiments/reports/emergency_ppo_learning_signal.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(
        f"# Emergency PPO learning signal\n\n"
        f"- latest_class: `{latest}`\n"
        f"- rows: {len(rows)}\n"
        f"- policy: never stop for WEAK/STAGNATION; only NUMERICAL_FAILURE\n"
        f"- updated: {doc['updated_at']}\n",
        encoding="utf-8",
    )
    print(json.dumps({"latest_class": latest, "n": len(classifications)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
