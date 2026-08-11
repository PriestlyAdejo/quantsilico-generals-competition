"""FINAL_SELECTION at hard end — refresh EMERGENCY_UPLOAD_THIS.md (no upload)."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    fallback = json.loads(
        (ROOT / "experiments/manifests/emergency_baseline_fallback.json").read_text(encoding="utf-8")
    )
    learned = None
    lp = ROOT / "experiments/manifests/emergency_learned_package_v1.json"
    qual = ROOT / "experiments/manifests/emergency_learned_qualification_v1.json"
    if lp.exists():
        learned = json.loads(lp.read_text(encoding="utf-8"))
    q = json.loads(qual.read_text(encoding="utf-8")) if qual.exists() else {}
    canary = json.loads(
        (ROOT / "experiments/manifests/emergency_rolling_ckpt_canary.json").read_text(encoding="utf-8")
    ) if (ROOT / "experiments/manifests/emergency_rolling_ckpt_canary.json").exists() else {}
    shadow = json.loads(
        (ROOT / "experiments/manifests/emergency_shadow_controls.json").read_text(encoding="utf-8")
    ) if (ROOT / "experiments/manifests/emergency_shadow_controls.json").exists() else {}

    # Priority: normal/learned with strongest paired evidence → established fault-free
    selected = None
    status = None
    if learned and q.get("TECHNICALLY_QUALIFIED"):
        # competitively unconfirmed but technically valid preferred over heuristic if exists
        # Plan priority: technically valid ZIP > learned attempt > ...
        # Final selection: normal → strongest paired → fault-free if inconclusive
        # Without paired Hunter wins, keep heuristic as upload pointer unless learned is TECHNICALLY_QUALIFIED
        # and we want learned attempt visible — plan says refresh with concrete ZIP.
        # Prefer learned technically qualified as selected learned candidate, but if no competitive
        # evidence, final upload insurance may remain heuristic.
        latest = (canary.get("evaluated") or [{}])[-1]
        has_hunter_wd = bool(latest.get("W") or latest.get("D"))
        if has_hunter_wd:
            selected = {
                "kind": "LEARNED_V1",
                "package_id": learned.get("package_id"),
                "zip": learned.get("package_path"),
                "sha256": learned.get("sha256"),
                "control_mode": "CONTROL_OFF",
                "reason": "technically_qualified_learned_with_canary_signal",
            }
            status = "EMERGENCY_LEARNED_PROVISIONAL_PACKAGE_EXISTS"
        else:
            # still record learned exists but select heuristic for upload insurance
            selected = {
                "kind": "HEURISTIC_FALLBACK_INSURANCE",
                "package_id": fallback.get("package_id")
                or fallback.get("heuristic_version")
                or "heuristic_v2_preppo",
                "zip": fallback.get("package_path")
                or fallback.get("zip")
                or fallback.get("package_zip_repo"),
                "sha256": fallback.get("sha256") or fallback.get("package_sha256"),
                "control_mode": "CONTROL_OFF",
                "reason": "learned_exists_but_competitively_unconfirmed_no_hunter_W_D; keep insurance",
                "learned_alternate": {
                    "zip": learned.get("package_path"),
                    "sha256": learned.get("sha256"),
                    "technical": learned.get("technical"),
                },
            }
            status = "EMERGENCY_BASELINE_FALLBACK_PACKAGE_EXISTS"
    else:
        fb_zip = (
            fallback.get("package_path")
            or fallback.get("zip")
            or fallback.get("package_zip_repo")
        )
        fb_sha = fallback.get("sha256") or fallback.get("package_sha256")
        selected = {
            "kind": "HEURISTIC_FALLBACK_INSURANCE",
            "package_id": fallback.get("package_id")
            or fallback.get("heuristic_version")
            or "heuristic_v2_preppo",
            "zip": fb_zip,
            "sha256": fb_sha,
            "control_mode": "CONTROL_OFF",
            "reason": "no_technically_qualified_learned_package",
            "learned_alternate": {
                "zip": (learned or {}).get("package_path"),
                "sha256": (learned or {}).get("sha256"),
                "technical": (learned or {}).get("technical") or q.get("TECHNICALLY_QUALIFIED"),
                "status_class": (learned or {}).get("status_class"),
                "latency_note": (q.get("latency_bench_linux_local") or {}),
            }
            if learned
            else None,
        }
        status = "EMERGENCY_BASELINE_FALLBACK_PACKAGE_EXISTS"

    # resolve zip path
    zip_path = Path(selected["zip"]) if selected.get("zip") else None
    if zip_path and not zip_path.is_absolute():
        zip_path = ROOT / zip_path
    if zip_path and zip_path.exists() and not selected.get("sha256"):
        selected["sha256"] = _sha256(zip_path)

    final = {
        "schema_version": 1,
        "kind": "EMERGENCY_FINAL_SELECTION",
        "status": status,
        "selected": selected,
        "shadow": {
            "replay_status": shadow.get("status"),
            "mpc": shadow.get("mpc") or "MPC_DEFERRED_DEADLINE_PROTECTION",
            "policy_benefit": "NOT_CLAIMED",
        },
        "canary_summary": [
            {"name": e.get("name"), "W": e.get("W"), "D": e.get("D"), "L": e.get("L"), "update": e.get("update")}
            for e in (canary.get("evaluated") or [])[-5:]
        ],
        "limitations": [
            "MANUAL_UPLOAD_ONLY",
            "no portal mutation from this gate",
            "COMPETITIVELY_UNCONFIRMED unless paired evidence present",
        ],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(ROOT / "experiments/manifests/emergency_final_selection.json", final)

    md = ROOT / "submission" / "EMERGENCY_UPLOAD_THIS.md"
    alt = selected.get("learned_alternate")
    md.write_text(
        "# EMERGENCY UPLOAD POINTER (manual only)\n\n"
        f"- **status**: `{status}`\n"
        f"- **selected_kind**: `{selected.get('kind')}`\n"
        f"- **ZIP**: `{selected.get('zip')}`\n"
        f"- **SHA-256**: `{selected.get('sha256')}`\n"
        f"- **control_mode**: `{selected.get('control_mode')}`\n"
        f"- **reason**: {selected.get('reason')}\n"
        + (
            f"- **learned_alternate_ZIP**: `{alt.get('zip')}`\n- **learned_alternate_SHA**: `{alt.get('sha256')}`\n"
            if alt
            else ""
        )
        + f"- **p99_ms**: `{q.get('p99_ms')}`\n"
        f"- **canary**: `{final['canary_summary']}`\n"
        f"- **MPC**: `MPC_DEFERRED_DEADLINE_PROTECTION`\n"
        f"- **upload**: MANUAL_UPLOAD_ONLY — do not auto-upload\n"
        f"- **updated_at**: {final['updated_at']}\n",
        encoding="utf-8",
    )
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
