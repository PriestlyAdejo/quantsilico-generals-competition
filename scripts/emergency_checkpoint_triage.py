"""CPU offline triage of R-E.6 checkpoints (finite/legal/COMPLETE)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path.home() / "quantsilico-runtime" / "emergency_rolling_v1"

CANDIDATES = [
    ("u1", ROOT / "experiments/competition_native_jax/v4_3_r_e6/ckpt_u1"),
    ("u152", ROOT / "experiments/competition_native_jax/v4_3_r_e6/ckpt_u152"),
    ("u304", ROOT / "experiments/competition_native_jax/v4_3_r_e6/ckpt_u304"),
    ("final", ROOT / "experiments/competition_native_jax/v4_3_r_e6/ckpt_final"),
]


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


def _check_npz(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "finite": False}
    try:
        data = np.load(path, allow_pickle=False)
        finite = True
        n = 0
        for k in data.files:
            arr = data[k]
            n += arr.size
            if not np.isfinite(arr).all():
                finite = False
                break
        return {"exists": True, "finite": finite, "arrays": len(data.files), "elements": int(n)}
    except Exception as e:
        return {"exists": True, "finite": False, "error": str(e)}


def main() -> int:
    rows = []
    for tag, ckpt in CANDIDATES:
        meta_path = ckpt / "meta.json"
        complete = (ckpt / "COMPLETE").exists()  # R-E.6 predate COMPLETE; treat meta+opt as publish
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        raw = _check_npz(ckpt / "raw.npz")
        ema = _check_npz(ckpt / "ema.npz")
        opt = _check_npz(ckpt / "opt_state.npz")
        legacy_ok = raw["finite"] and ema["finite"] and opt["finite"] and meta_path.exists()
        consumable = complete or legacy_ok  # grandfather R-E.6 parents
        for which in ("raw", "ema"):
            rows.append(
                {
                    "id": f"ckpt_{tag}_{which}",
                    "tag": tag,
                    "which": which,
                    "path": str(ckpt.relative_to(ROOT)).replace("\\", "/"),
                    "update": meta.get("update"),
                    "transitions": meta.get("transitions"),
                    "COMPLETE_marker": complete,
                    "legacy_r_e6_publish": legacy_ok and not complete,
                    "consumable": consumable,
                    "raw": raw,
                    "ema": ema,
                    "opt_state": opt,
                    "learner": (meta.get("lineage") or {}).get("learner_implementation_hash"),
                }
            )

    report = {
        "schema_version": 1,
        "kind": "EMERGENCY_CHECKPOINT_TRIAGE",
        "status": "COMPLETE",
        "candidates": rows,
        "consumable_count": sum(1 for r in rows if r["consumable"]),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(ROOT / "experiments/manifests/emergency_checkpoint_triage.json", report)
    _atomic_write_json(RUNTIME / "eval" / "checkpoint_triage.json", report)
    print(json.dumps({"status": "COMPLETE", "consumable": report["consumable_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
