"""IMMEDIATE_BASELINE_PACKAGE_FALLBACK — verify strongest known runnable ZIP."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path.home() / "quantsilico-runtime" / "emergency_rolling_v1"

# Strongest established portal-qualified heuristic (already submitted).
CANDIDATES = [
    {
        "id": "QS-PUBLIC-V001_heuristic_v2_preppo",
        "policy_source": "heuristic_v2f_plus_planner_terminal_form",
        "heuristic_version": "heuristic_v2_preppo_8f7405fe9834161c",
        "already_submitted": True,
        "zip_path": ROOT / "submission/packages/heuristic_v2_preppo_8f7405fe9834161c_packaged.zip",
        "alt_zip": ROOT / "submission/packages/QS-PUBLIC-V001/e1237f77dee46993/package.zip",
        "expected_sha256": "e1237f77dee469935fc3a60811b9a34522b83dd37bf4d76fa2555e6107a8edfa",
        "linux_parity": "PASS",
        "portal_verdict": "QUALIFIED",
        "report": ROOT / "submission/packages/heuristic_v2_preppo_8f7405fe9834161c_packaged.report.json",
    },
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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    chosen = None
    checks = []
    for c in CANDIDATES:
        zip_path = c["zip_path"] if c["zip_path"].exists() else c.get("alt_zip")
        if zip_path is None or not Path(zip_path).exists():
            checks.append({"id": c["id"], "status": "MISSING_ZIP"})
            continue
        sha = _sha256(Path(zip_path))
        ok = sha.lower() == c["expected_sha256"].lower()
        entry = {
            "id": c["id"],
            "zip_path": str(Path(zip_path).relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha,
            "expected_sha256": c["expected_sha256"],
            "sha_match": ok,
            "linux_parity": c["linux_parity"],
            "portal_verdict": c["portal_verdict"],
            "already_submitted": c["already_submitted"],
            "status": "VERIFIED" if ok else "SHA_MISMATCH",
        }
        checks.append(entry)
        if ok and chosen is None:
            chosen = {**c, **entry, "absolute_zip": str(Path(zip_path).resolve())}

    if chosen is None:
        report = {
            "schema_version": 1,
            "kind": "IMMEDIATE_BASELINE_PACKAGE_FALLBACK",
            "status": "NO_TECHNICALLY_VALID_PACKAGE",
            "checks": checks,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write_json(ROOT / "experiments/manifests/emergency_baseline_fallback.json", report)
        return 1

    # Copy immutable fallback into emergency packages dir (do not rebuild)
    dest_dir = RUNTIME / "packages" / "baseline_fallback" / chosen["sha256"][:16]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_zip = dest_dir / "package.zip"
    if not dest_zip.exists():
        shutil.copy2(chosen["absolute_zip"], dest_zip)
    # verify copy
    copy_sha = _sha256(dest_zip)
    assert copy_sha.lower() == chosen["sha256"].lower()

    identity = {
        "schema_version": 1,
        "kind": "EMERGENCY_BASELINE_FALLBACK_PACKAGE",
        "status": "EMERGENCY_BASELINE_FALLBACK_PACKAGE_EXISTS",
        "classification": [
            "EMERGENCY_BASELINE_FALLBACK_PACKAGE",
            "TECHNICALLY_QUALIFIED",
            "NOT_A_V4_3_LEARNED_CANDIDATE",
            "MANUAL_UPLOAD_ONLY",
        ],
        "policy_source": chosen["policy_source"],
        "heuristic_version": chosen["heuristic_version"],
        "already_submitted": chosen["already_submitted"],
        "package_sha256": chosen["sha256"],
        "package_zip_repo": chosen["zip_path"],
        "package_zip_runtime": str(dest_zip),
        "linux_parity": chosen["linux_parity"],
        "portal_verdict": chosen["portal_verdict"],
        "technical_qualification": "VERIFIED_EXISTING_LINUX_PASS_AND_PORTAL_QUALIFIED",
        "requalification_mode": "VERIFY_NOT_REBUILD",
        "checks": checks,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(ROOT / "experiments/manifests/emergency_baseline_fallback.json", identity)
    _atomic_write_json(RUNTIME / "packages" / "baseline_fallback.json", identity)

    roles = ROOT / "submission/roles"
    roles.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        roles / "emergency_candidate.json",
        {
            "schema_version": 1,
            "role": "emergency_candidate",
            "status": "EMERGENCY_BASELINE_FALLBACK_PACKAGE_EXISTS",
            "fallback_artefact": identity,
            "learned_candidate": None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    # Update programme pointer
    prog_path = ROOT / "experiments/manifests/emergency_rolling_programme_state.json"
    if prog_path.exists():
        prog = json.loads(prog_path.read_text(encoding="utf-8"))
        prog["baseline_fallback"] = "experiments/manifests/emergency_baseline_fallback.json"
        prog["package_status"] = "EMERGENCY_BASELINE_FALLBACK_PACKAGE_EXISTS"
        prog["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(prog_path, prog)

    print(json.dumps({"status": identity["status"], "sha256": chosen["sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
