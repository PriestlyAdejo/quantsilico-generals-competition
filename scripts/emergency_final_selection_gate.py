"""FINAL_EMERGENCY_PACKAGE_SELECTION_GATE + EMERGENCY_UPLOAD_THIS.md."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

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


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    candidates = []

    # Normal qualified
    rec = _load(ROOT / "submission/roles/recommended.json") or {}
    if rec.get("package_zip") and rec.get("status") not in (
        None,
        "NO_CANDIDATE_CURRENTLY_RECOMMENDED",
    ):
        candidates.append(
            {
                "class": "NORMAL_UPLOAD_READY_CANDIDATE",
                "package_zip": rec.get("package_zip"),
                "package_sha256": rec.get("package_sha256"),
                "cpu_p99": None,
                "canary_score": None,
                "faults": [],
                "established": True,
            }
        )

    # Learned emergency
    learned = _load(ROOT / "experiments/manifests/emergency_learned_package.json")
    if learned and learned.get("status") == "EMERGENCY_LEARNED_PROVISIONAL_PACKAGE_EXISTS":
        candidates.append(
            {
                "class": "EMERGENCY_LEARNED_PROVISIONAL",
                "package_zip": learned.get("package_zip"),
                "package_sha256": learned.get("package_sha256"),
                "cpu_p99": learned.get("cpu_p99"),
                "canary_score": learned.get("canary_score"),
                "faults": learned.get("faults") or [],
                "established": False,
            }
        )

    # Controlled
    controlled = _load(ROOT / "experiments/manifests/emergency_controlled_package.json")
    if controlled and controlled.get("technically_qualified"):
        candidates.append(
            {
                "class": "EMERGENCY_CONTROLLED",
                "package_zip": controlled.get("package_zip"),
                "package_sha256": controlled.get("package_sha256"),
                "cpu_p99": controlled.get("cpu_p99"),
                "canary_score": controlled.get("canary_score"),
                "faults": controlled.get("faults") or [],
                "established": False,
            }
        )

    # Baseline fallback
    base = _load(ROOT / "experiments/manifests/emergency_baseline_fallback.json")
    if base and base.get("status") == "EMERGENCY_BASELINE_FALLBACK_PACKAGE_EXISTS":
        candidates.append(
            {
                "class": "EMERGENCY_BASELINE_FALLBACK",
                "package_zip": base.get("package_zip_repo"),
                "package_sha256": base.get("package_sha256"),
                "cpu_p99": None,
                "canary_score": None,
                "faults": [],
                "established": True,
                "already_submitted": base.get("already_submitted"),
                "NOT_A_V4_3_LEARNED_CANDIDATE": True,
            }
        )

    selected = None
    rejection: list[dict] = []
    # Priority 1: normal
    normals = [c for c in candidates if c["class"] == "NORMAL_UPLOAD_READY_CANDIDATE"]
    if normals:
        selected = normals[0]
        for c in candidates:
            if c is not selected:
                rejection.append({**c, "rejection_reason": "NORMAL_CANDIDATE_PREFERRED"})
    else:
        # Priority 2: strongest paired emergency evidence among technically valid
        scored = []
        for c in candidates:
            if c.get("faults"):
                rejection.append({**c, "rejection_reason": "HAS_FAULTS"})
                continue
            score = c.get("canary_score")
            scored.append((score is not None, float(score or 0.0), c.get("established", False), c))
        if scored:
            # Prefer higher canary when present; if inconclusive (all None), prefer established
            with_scores = [x for x in scored if x[0]]
            if with_scores:
                with_scores.sort(key=lambda t: t[1], reverse=True)
                selected = with_scores[0][3]
            else:
                established = [x[3] for x in scored if x[2]]
                selected = established[0] if established else scored[0][3]
            for _hs, _sc, _est, c in scored:
                if c is not selected:
                    rejection.append(
                        {
                            **c,
                            "rejection_reason": (
                                "WEAKER_OR_INCONCLUSIVE_VS_SELECTED"
                                if not c.get("established")
                                else "NOT_SELECTED_LEARNED_NOT_STRONGER"
                            ),
                        }
                    )

    if selected is None:
        status = "NO_TECHNICALLY_VALID_PACKAGE"
        gate = {
            "schema_version": 1,
            "kind": "FINAL_EMERGENCY_PACKAGE_SELECTION_GATE",
            "status": status,
            "candidates": candidates,
            "selected": None,
            "rejected": rejection,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write_json(ROOT / "experiments/manifests/final_emergency_package_selection_gate.json", gate)
        md = ROOT / "submission/EMERGENCY_UPLOAD_THIS.md"
        md.write_text(
            "# EMERGENCY_UPLOAD_THIS\n\nStatus: **NO_TECHNICALLY_VALID_PACKAGE**\n\nDo not upload.\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": status}, indent=2))
        return 1

    status_map = {
        "NORMAL_UPLOAD_READY_CANDIDATE": "NORMAL_UPLOAD_READY_CANDIDATE_EXISTS",
        "EMERGENCY_LEARNED_PROVISIONAL": "EMERGENCY_LEARNED_PROVISIONAL_PACKAGE_EXISTS",
        "EMERGENCY_CONTROLLED": "EMERGENCY_LEARNED_PROVISIONAL_PACKAGE_EXISTS",
        "EMERGENCY_BASELINE_FALLBACK": "EMERGENCY_BASELINE_FALLBACK_PACKAGE_EXISTS",
    }
    final_status = status_map[selected["class"]]
    reason = {
        "NORMAL_UPLOAD_READY_CANDIDATE": "Normal protocol-qualified candidate exists.",
        "EMERGENCY_LEARNED_PROVISIONAL": "Strongest paired emergency evidence among learned packages.",
        "EMERGENCY_CONTROLLED": "Controlled package strongest with two-verdict pass.",
        "EMERGENCY_BASELINE_FALLBACK": (
            "No stronger learned/controlled evidence; retain established fault-free fallback "
            "(not preferred merely for being older — learned was not demonstrably better)."
        ),
    }[selected["class"]]

    gate = {
        "schema_version": 1,
        "kind": "FINAL_EMERGENCY_PACKAGE_SELECTION_GATE",
        "status": "PASSED",
        "final_status": final_status,
        "selected": {**selected, "selection_reason": reason},
        "rejected": rejection,
        "all_candidates": candidates,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(ROOT / "experiments/manifests/final_emergency_package_selection_gate.json", gate)

    md = ROOT / "submission/EMERGENCY_UPLOAD_THIS.md"
    md.write_text(
        "\n".join(
            [
                "# EMERGENCY_UPLOAD_THIS",
                "",
                f"Status: **{final_status}**",
                "",
                f"- Package ZIP: `{selected['package_zip']}`",
                f"- Package SHA-256: `{selected['package_sha256']}`",
                f"- Class: `{selected['class']}`",
                f"- Selection reason: {reason}",
                "",
                "Manual upload only. Portal mutation / automatic upload **not** authorised.",
                "",
                "Gate: `experiments/manifests/final_emergency_package_selection_gate.json`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    prog_path = ROOT / "experiments/manifests/emergency_rolling_programme_state.json"
    if prog_path.exists():
        prog = json.loads(prog_path.read_text(encoding="utf-8"))
        prog["final_status"] = final_status
        prog["emergency_upload_this"] = "submission/EMERGENCY_UPLOAD_THIS.md"
        prog["selection_gate"] = "experiments/manifests/final_emergency_package_selection_gate.json"
        prog["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(prog_path, prog)

    print(
        json.dumps(
            {
                "status": final_status,
                "sha256": selected["package_sha256"],
                "zip": selected["package_zip"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
