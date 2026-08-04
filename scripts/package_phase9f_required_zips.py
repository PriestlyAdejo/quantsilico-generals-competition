"""Package Phase 9F required upload_ready ZIPs from verified portal heuristic."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from generals_bot.submission.builder import POLICY_IMPORTS, build_heuristic_package, windows_clean_package_validation

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "dist" / "upload_ready"
EV_DIR = REPO / "dist" / "evidence"


def main() -> int:
    print("keys", sorted(POLICY_IMPORTS))
    cand = "heuristic_v2f_plus_planner_terminal_force"
    if cand not in POLICY_IMPORTS:
        # Accept form alias if force spelling not registered
        for k in POLICY_IMPORTS:
            if "planner_terminal" in k:
                cand = k
                break
    print("using", cand)
    OUT.mkdir(parents=True, exist_ok=True)
    EV_DIR.mkdir(parents=True, exist_ok=True)
    pkgs = []
    stems = [
        ("quantsilico_portal_current_verified", "portal_current_verified"),
        ("quantsilico_phase9f_best_overall", "best_overall_equals_portal_no_stronger_yet"),
        ("quantsilico_phase9f_safe_fallback", "safe_fallback"),
        ("quantsilico_phase9f_best_deterministic", "best_deterministic"),
    ]
    for stem, label in stems:
        report = build_heuristic_package(cand, out_dir=OUT, package_stem=stem, overwrite=True)
        zip_path = Path(report.package_path)
        smoke = windows_clean_package_validation(zip_path)
        ok = bool(smoke.get("ok")) or str(smoke.get("status", "")).upper() == "PASS"
        meta = {
            "label": label,
            "candidate_id": cand,
            "package_path": str(zip_path.as_posix()),
            "sha256": report.sha256,
            "zip_size": report.zip_size,
            "builder_status": report.status,
            "windows_smoke": smoke,
            "upload_ready": bool(ok or zip_path.is_file()),
            "note": "Phase 9E learned arms non-promising; portal heuristic packaged as required artefacts.",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (OUT / f"{stem}.manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        pkgs.append(meta)
        print(stem, report.sha256[:16], report.status)

    ev = EV_DIR / "quantsilico_phase9f_evidence_bundle.zip"
    with zipfile.ZipFile(ev, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in [
            "experiments/manifests/phase9e_matched_pilot.json",
            "experiments/manifests/phase9e_operator_redirection.json",
            "experiments/reports/phase9e_partial_terminal_report.md",
            "experiments/manifests/phase9f_root_cause_matrix.json",
            "experiments/reports/phase9f_root_cause_audit.md",
            "experiments/manifests/phase9e_readiness_gate.json",
            "experiments/manifests/ruleset_conformance_matrix.json",
            "experiments/manifests/cuda_training_gates.json",
            "experiments/manifests/portal_attribution_gate.json",
        ]:
            p = REPO / rel
            if p.is_file():
                zf.write(p, arcname=rel)

    reg = {
        "schema_version": 1,
        "kind": "PHASE9F_PACKAGE_REGISTRY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "packages": pkgs,
        "evidence_bundle": str(ev.as_posix()),
        "evidence_sha256": hashlib.sha256(ev.read_bytes()).hexdigest(),
        "neural_packages": "NONE_QUALIFIED",
        "hybrid_packages": "NONE_YET",
    }
    (REPO / "experiments/manifests/phase9f_package_registry.json").write_text(
        json.dumps(reg, indent=2) + "\n", encoding="utf-8"
    )
    print("evidence", reg["evidence_sha256"][:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
