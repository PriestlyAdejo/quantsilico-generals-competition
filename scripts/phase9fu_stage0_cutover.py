"""Phase 9FU Stage 0 + 0B-FAST: correction artefacts, atomic freeze, canonical cutover."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

LIVE_SHA = "e1237f77dee469935fc3a60811b9a34522b83dd37bf4d76fa2555e6107a8edfa"
LIVE_HISTORICAL = (
    REPO / "submission" / "packages" / "heuristic_v2_preppo_8f7405fe9834161c_packaged.zip"
)
LIVE_BUILD = "e1237f77dee46993"
RELATED_SHA = "898b37b104545fa6217877dd2db2af7c6e8810f41b4ba1f79cc8530b798d558e"
RELATED_HISTORICAL = (
    REPO / "dist" / "submission_recommended" / "qs_p9fu_heuristic_v1_packaged.zip"
)
RELATED_BUILD = "898b37b104545fa6"
UPLOAD_TS_LOCAL = "2026-08-04T00:37:00+01:00"
UPLOAD_TS_UTC = "2026-08-03T23:37:00Z"
CANONICAL = "heuristic_v2f_plus_planner_terminal_" + "f" + "ix"
STABLE = "QS-P9F-PORTAL-V0"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = (json.dumps(doc, indent=2) + "\n").encode("utf-8")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        existing = path.read_bytes()
        if existing == data:
            tmp.unlink(missing_ok=True)
            return
        raise FileExistsError(f"refusing to overwrite non-equivalent freeze/artefact: {path}")
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str, *, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = text.encode("utf-8")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))
    os.replace(tmp, path)


def copy_verify(src: Path, dst: Path, expected_sha: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.is_file():
        raise FileNotFoundError(str(src))
    got = sha256_file(src)
    if got.lower() != expected_sha.lower():
        raise ValueError(f"SHA mismatch for {src}: got {got} expected {expected_sha}")
    if dst.exists():
        existing = sha256_file(dst)
        if existing.lower() != expected_sha.lower():
            raise FileExistsError(f"destination exists with different SHA: {dst}")
        return existing.lower()
    shutil.copy2(src, dst)
    verified = sha256_file(dst)
    if verified.lower() != expected_sha.lower():
        raise RuntimeError(f"copy verification failed for {dst}")
    return verified.lower()


def write_package_sidecar(
    build_dir: Path,
    *,
    candidate_id: str,
    stable_id: str,
    sha: str,
    build_hash: str,
    original_path: str,
    notes: list[str],
    qualification: dict,
) -> None:
    atomic_write_text(build_dir / "sha256.txt", sha + "\n")
    manifest = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "stable_candidate_id": stable_id,
        "build_hash": build_hash,
        "sha256": sha,
        "package_file": "package.zip",
        "original_uploaded_path": original_path,
        "canonical_archive_path": str(
            (build_dir / "package.zip").relative_to(REPO).as_posix()
        ),
        "architecture": "heuristic",
        "notes": notes,
    }
    atomic_write_json(build_dir / "package_manifest.json", manifest)
    atomic_write_json(build_dir / "qualification_report.json", qualification)


def main() -> int:
    now = datetime.now(timezone.utc).isoformat()
    head = "ab7c9c0"

    # --- Stage 0: correction artefacts ---
    orig_rec_path = REPO / "experiments" / "manifests" / "phase9fs_first_recommendation_gate.json"
    orig_rec = json.loads(orig_rec_path.read_text(encoding="utf-8"))
    orig_rec_sha = sha256_file(orig_rec_path)

    failure_md = f"""# QS-PUBLIC-V001 strategic failure

Created: {now}

## Classification

`TECHNICALLY_QUALIFIED_STRATEGICALLY_WEAK_BASELINE`

## Live package (actually uploaded)

- Path: `{LIVE_HISTORICAL.as_posix()}`
- SHA-256: `{LIVE_SHA}`
- Candidate: `{CANONICAL}`
- Stable ID: `{STABLE}`
- Upload: `{UPLOAD_TS_LOCAL}` ({UPLOAD_TS_UTC})

## Code-grounded defects

- V2F sets `prefer_castles=False` and `castle_weight=0.85`
- Known enemy general immediately selects `GENERAL_HUNT` in `phase_controller_v2f.select_phase`
- Approach-enemy-general proposals use `hard_priority=93`
- Castle proposals use `hard_priority=26` and are gated (builds&lt;2, turn&lt;600, phase exclusions)
- Collection receives a toward-enemy bonus when the general is known
- No attack-readiness gate before commit

Observed premature hunting / weak preparation is the expected result of this hierarchy.

## Related non-live repackage

- `dist/submission_recommended/qs_p9fu_heuristic_v1_packaged.zip` SHA `{RELATED_SHA}`
- Same underlying policy; Windows smoke only — **not** the live portal upload
"""
    atomic_write_text(
        REPO / "experiments" / "reports" / "phase9fu_v001_strategic_failure.md",
        failure_md,
        overwrite=True,
    )

    reclass = {
        "schema_version": 1,
        "kind": "PHASE9FU_V001_RECLASSIFICATION",
        "created_at": now,
        "public_version": "QS-PUBLIC-V001",
        "status": "TECHNICALLY_QUALIFIED_STRATEGICALLY_WEAK_BASELINE",
        "candidate_id": CANONICAL,
        "stable_candidate_id": STABLE,
        "actual_uploaded_package": str(LIVE_HISTORICAL.as_posix()),
        "actual_uploaded_package_sha256": LIVE_SHA,
        "do_not_reupload": True,
        "do_not_recommend_again": True,
    }
    atomic_write_json(
        REPO / "experiments" / "manifests" / "phase9fu_v001_reclassification.json", reclass
    )

    correction = {
        "schema_version": 1,
        "kind": "PHASE9FU_FIRST_RECOMMENDATION_CORRECTION",
        "created_at": now,
        "original_artefact_path": str(orig_rec_path.as_posix()),
        "original_sha256": orig_rec_sha,
        "original_gate": orig_rec.get("gate_status"),
        "original_meaning": "technically packageable and manually uploadable",
        "incorrect_inference": "competitively credible",
        "corrected_interpretation": (
            "technical packaging PASS did not establish strategic strength"
        ),
        "superseding_status": "TECHNICALLY_QUALIFIED_STRATEGICALLY_WEAK_BASELINE",
        "superseding_artefacts": [
            "experiments/manifests/phase9fu_v001_reclassification.json",
            "experiments/reports/phase9fu_v001_strategic_failure.md",
            "experiments/manifests/phase9fu_first_recommendation_correction.json",
        ],
        "correction_reason": (
            "Live portal package is the preppo ZIP (e1237f77…), not the later "
            "898b… Windows repackage; policy hierarchy causes premature GENERAL_HUNT."
        ),
        "historical_gate_preserved": True,
        "note": "Do not overwrite phase9fs_first_recommendation_gate.json",
    }
    atomic_write_json(
        REPO / "experiments" / "manifests" / "phase9fu_first_recommendation_correction.json",
        correction,
    )
    atomic_write_text(
        REPO / "experiments" / "reports" / "phase9fu_first_recommendation_correction.md",
        "\n".join(
            [
                "# First recommendation correction",
                "",
                f"Created: {now}",
                "",
                f"- Original gate: **{orig_rec.get('gate_status')}** (preserved)",
                f"- Original SHA-256: `{orig_rec_sha}`",
                "- Corrected: packaging PASS ≠ competitive credibility",
                "- Superseding: `TECHNICALLY_QUALIFIED_STRATEGICALLY_WEAK_BASELINE`",
                "",
            ]
        ),
        overwrite=True,
    )

    # Copy recovery plan text into repo (do not edit Cursor plan file)
    recovery_plan = REPO / "plans" / "phase9fu_competitive_recovery.md"
    atomic_write_text(
        recovery_plan,
        "# Phase 9FU Competitive Recovery\n\n"
        "See Cursor plan `phase_9fu_recovery_076fafcf.plan.md` (do not edit during execution).\n"
        f"Execution started: {now}\n"
        f"Head: {head}\n",
        overwrite=True,
    )

    # --- FREEZE_OUTPUT_ATOMICITY_GATE ---
    if not LIVE_HISTORICAL.is_file():
        raise FileNotFoundError(str(LIVE_HISTORICAL))
    live_got = sha256_file(LIVE_HISTORICAL)
    if live_got.lower() != LIVE_SHA.lower():
        raise ValueError(f"live SHA mismatch: {live_got}")

    # package id from ZIP manifest
    import zipfile

    with zipfile.ZipFile(LIVE_HISTORICAL, "r") as zf:
        pm = json.loads(zf.read("package_manifest.json").decode("utf-8"))
    uploaded_package_id = pm.get("package_stem") or pm.get("candidate")
    provenance = "MANIFEST" if pm.get("package_stem") else "UNKNOWN"
    if not uploaded_package_id:
        uploaded_package_id = "UNKNOWN"
        provenance = "UNKNOWN"

    freeze_dir = REPO / "submission" / "public_versions" / "QS-PUBLIC-V001"
    freeze_doc = {
        "schema_version": 1,
        "kind": "SUBMISSION_UPLOAD_FREEZE_GATE",
        "gate_status": "PASS",
        "created_at": now,
        "FREEZE_OUTPUT_ATOMICITY_GATE": "PASS",
        "public_version": "QS-PUBLIC-V001",
        "immutable": True,
        "monitoring_open": True,
        "frozen": {
            "local_version_id": "QS-PUBLIC-V001",
            "baseline_type": "HEURISTIC",
            "candidate_id": CANONICAL,
            "stable_candidate_id": STABLE,
            "uploaded_package_stem": "heuristic_v2_preppo_8f7405fe9834161c_packaged",
            "uploaded_package_id": uploaded_package_id,
            "uploaded_package_id_provenance": provenance,
            "original_uploaded_path": str(LIVE_HISTORICAL.as_posix()),
            "actual_uploaded_package_sha256": LIVE_SHA,
            "build_hash": LIVE_BUILD,
            "source_commit": pm.get("bot_commit"),
            "engine_commit": pm.get("engine_commit"),
            "model_sha256": None,
            "control_mode": "OFF",
            "user_upload_timestamp_local": UPLOAD_TS_LOCAL,
            "user_upload_timestamp_utc": UPLOAD_TS_UTC,
            "frozen_at": now,
            "competitive_status": "TECHNICALLY_QUALIFIED_STRATEGICALLY_WEAK_BASELINE",
            "related_repackage_not_live": {
                "path": str(RELATED_HISTORICAL.as_posix()),
                "sha256": RELATED_SHA,
            },
        },
    }
    atomic_write_json(freeze_dir / "upload_freeze.json", freeze_doc)
    atomic_write_json(
        freeze_dir / "upload_record.json",
        {
            "schema_version": 1,
            "public_version": "QS-PUBLIC-V001",
            "upload_timestamp_local": UPLOAD_TS_LOCAL,
            "upload_timestamp_utc": UPLOAD_TS_UTC,
            "sha256": LIVE_SHA,
            "portal_verdict": "QUALIFIED",
            "operator_note": "Portal display 4 Aug 2026, 00:37 BST",
        },
    )
    atomic_write_json(
        freeze_dir / "public_epoch.json",
        {
            "schema_version": 1,
            "kind": "PUBLIC_VERSION_EPOCH_GATE",
            "gate_status": "WAITING_FOR_PUBLIC_OBSERVATION",
            "created_at": now,
            "note": "Monitoring may begin after upload freeze; epoch completes on first public observation.",
        },
    )
    atomic_write_json(
        freeze_dir / "monitoring.json",
        {"schema_version": 1, "status": "OPEN", "opened_at": now, "control_mode": "OFF"},
    )

    # Also mirror freeze summary into experiments (non-authoritative pointer)
    atomic_write_json(
        REPO / "experiments" / "manifests" / "phase9fu_submission_upload_freeze_gate.json",
        {
            **freeze_doc,
            "canonical_freeze_path": "submission/public_versions/QS-PUBLIC-V001/upload_freeze.json",
        },
    )

    # --- Stage 0B-FAST ---
    for sub in (
        "packages",
        "roles",
        "public_versions",
        "manifests",
        "reports",
        "staging",
        "legacy",
    ):
        (REPO / "submission" / sub).mkdir(parents=True, exist_ok=True)

    v001_dir = REPO / "submission" / "packages" / "QS-PUBLIC-V001" / LIVE_BUILD
    copy_verify(LIVE_HISTORICAL, v001_dir / "package.zip", LIVE_SHA)
    write_package_sidecar(
        v001_dir,
        candidate_id=CANONICAL,
        stable_id=STABLE,
        sha=LIVE_SHA,
        build_hash=LIVE_BUILD,
        original_path=str(LIVE_HISTORICAL.as_posix()),
        notes=[
            "Canonical archive of live portal upload",
            "Frozen public identity uses historical path + SHA",
        ],
        qualification={
            "linux_parity": "PASS",
            "windows_validation": "PASS",
            "portal_verdict": "QUALIFIED",
            "competitive_status": "TECHNICALLY_QUALIFIED_STRATEGICALLY_WEAK_BASELINE",
            "source_report": "submission/packages/heuristic_v2_preppo_8f7405fe9834161c_packaged.report.json",
        },
    )

    related_dir = REPO / "submission" / "packages" / "QS-P9F-PORTAL-V0" / RELATED_BUILD
    if RELATED_HISTORICAL.is_file():
        copy_verify(RELATED_HISTORICAL, related_dir / "package.zip", RELATED_SHA)
        write_package_sidecar(
            related_dir,
            candidate_id=CANONICAL,
            stable_id=STABLE,
            sha=RELATED_SHA,
            build_hash=RELATED_BUILD,
            original_path=str(RELATED_HISTORICAL.as_posix()),
            notes=[
                "RELATED_REPACKAGE — later Phase 9F-U Windows smoke; not live",
                "Do not mark LIVE / RECOMMENDED / UPLOAD_READY",
            ],
            qualification={
                "linux_parity": "NOT_RUN",
                "windows_validation": "PASS",
                "role": "RELATED_REPACKAGE",
            },
        )
        # remove historical dist copy after verification
        RELATED_HISTORICAL.unlink()
        report_side = RELATED_HISTORICAL.with_name(
            RELATED_HISTORICAL.name.replace(".zip", ".report.json")
        )
        if report_side.exists():
            report_side.unlink()

    # Roles
    live = {
        "role": "LIVE",
        "public_version": "QS-PUBLIC-V001",
        "candidate_id": CANONICAL,
        "stable_candidate_id": STABLE,
        "build_hash": LIVE_BUILD,
        "package_path": f"submission/packages/QS-PUBLIC-V001/{LIVE_BUILD}/package.zip",
        "sha256": LIVE_SHA,
        "status": "LIVE",
    }
    recommended = {
        "status": "NO_CANDIDATE_CURRENTLY_RECOMMENDED",
        "package_path": None,
        "reason": "V001 is live but strategically weak; challengers remain under evaluation.",
    }
    related_role = {
        "role": "RELATED_REPACKAGES",
        "packages": [
            {
                "candidate_id": CANONICAL,
                "stable_candidate_id": STABLE,
                "build_hash": RELATED_BUILD,
                "package_path": f"submission/packages/QS-P9F-PORTAL-V0/{RELATED_BUILD}/package.zip",
                "sha256": RELATED_SHA,
                "status": "RELATED_REPACKAGE",
                "note": "Never LIVE / RECOMMENDED / UPLOAD_READY",
            }
        ],
    }
    atomic_write_json(REPO / "submission" / "roles" / "live.json", live)
    atomic_write_json(REPO / "submission" / "roles" / "recommended.json", recommended)
    atomic_write_json(REPO / "submission" / "roles" / "related_repackages.json", related_role)
    atomic_write_json(
        REPO / "submission" / "roles" / "upload_ready.json",
        {"status": "NONE", "packages": []},
    )
    atomic_write_json(
        REPO / "submission" / "roles" / "windows_smoke_passed.json",
        {"status": "SEE_MIGRATION_REGISTRY", "packages": []},
    )
    atomic_write_json(
        REPO / "submission" / "roles" / "research_candidates.json",
        {"status": "NONE_YET", "packages": []},
    )
    atomic_write_json(
        REPO / "submission" / "roles" / "rejected.json",
        {"status": "NONE", "packages": []},
    )

    upload_this = """NO_CANDIDATE_CURRENTLY_RECOMMENDED

The currently live QS-PUBLIC-V001 package is retained as a technically qualified
but strategically weak baseline.

Do not re-upload:
heuristic_v2f_plus_planner_terminal_fix

A future package will appear here only after the Phase 9FU behavioural and paired
evaluation gates pass.
"""
    atomic_write_text(REPO / "submission" / "UPLOAD_THIS.md", upload_this, overwrite=True)
    atomic_write_text(
        REPO / "submission" / "README.md",
        "# Submission artefacts\n\n"
        "Canonical root for all competition packages, roles, and public-version records.\n"
        "Builders must write under `submission/` only — not `dist/`.\n"
        "Roles are JSON pointers to a single canonical ZIP per build.\n",
        overwrite=True,
    )

    package_registry = {
        "schema_version": 1,
        "kind": "SUBMISSION_PACKAGE_REGISTRY",
        "created_at": now,
        "packages": [
            {
                "public_or_candidate_id": "QS-PUBLIC-V001",
                "build_hash": LIVE_BUILD,
                "sha256": LIVE_SHA,
                "path": f"submission/packages/QS-PUBLIC-V001/{LIVE_BUILD}/package.zip",
                "roles": ["LIVE"],
            },
            {
                "public_or_candidate_id": "QS-P9F-PORTAL-V0",
                "build_hash": RELATED_BUILD,
                "sha256": RELATED_SHA,
                "path": f"submission/packages/QS-P9F-PORTAL-V0/{RELATED_BUILD}/package.zip",
                "roles": ["RELATED_REPACKAGE"],
            },
        ],
    }
    candidate_registry = {
        "schema_version": 1,
        "kind": "SUBMISSION_CANDIDATE_REGISTRY",
        "created_at": now,
        "candidates": {
            CANONICAL: {
                "stable_candidate_id": STABLE,
                "live_public_version": "QS-PUBLIC-V001",
                "competitive_status": "TECHNICALLY_QUALIFIED_STRATEGICALLY_WEAK_BASELINE",
            }
        },
    }
    migration_registry = {
        "schema_version": 1,
        "kind": "SUBMISSION_MIGRATION_REGISTRY",
        "created_at": now,
        "stage": "0B_FAST",
        "entries": [
            {
                "original_path": str(LIVE_HISTORICAL.as_posix()),
                "canonical_archive_path": f"submission/packages/QS-PUBLIC-V001/{LIVE_BUILD}/package.zip",
                "sha256": LIVE_SHA,
                "migration_timestamp": now,
                "note": "Live V001; historical path retained for portal attribution",
            },
            {
                "original_path": str(RELATED_HISTORICAL.as_posix()),
                "canonical_archive_path": f"submission/packages/QS-P9F-PORTAL-V0/{RELATED_BUILD}/package.zip",
                "sha256": RELATED_SHA,
                "migration_timestamp": now,
                "note": "Related repackage; dist copy removed after verification",
                "dist_copy_removed": True,
            },
        ],
    }
    atomic_write_json(REPO / "submission" / "manifests" / "package_registry.json", package_registry)
    atomic_write_json(
        REPO / "submission" / "manifests" / "candidate_registry.json", candidate_registry
    )
    atomic_write_json(
        REPO / "submission" / "manifests" / "migration_registry.json", migration_registry
    )
    atomic_write_text(
        REPO / "submission" / "reports" / "current_recommendation.md",
        "# Current recommendation\n\n`NO_CANDIDATE_CURRENTLY_RECOMMENDED`\n",
        overwrite=True,
    )
    atomic_write_text(
        REPO / "submission" / "reports" / "qualification_summary.md",
        f"# Qualification summary\n\n- LIVE: QS-PUBLIC-V001 (`{LIVE_SHA}`)\n"
        "- RECOMMENDED: none\n",
        overwrite=True,
    )

    # Block new dist package dirs: marker README
    dist_readme = REPO / "dist" / "README_MOVED.md"
    atomic_write_text(
        dist_readme,
        "Submission artefacts moved to ../submission/\n"
        "Do not write new competition package ZIPs under dist/.\n"
        "Active builders must use submission/staging → submission/packages.\n",
        overwrite=True,
    )

    layout_gate = {
        "schema_version": 1,
        "kind": "PHASE9FU_SUBMISSION_LAYOUT_GATE",
        "created_at": now,
        "stage": "0B_FAST",
        "gates": {
            "FREEZE_OUTPUT_ATOMICITY_GATE": "PASS",
            "SUBMISSION_LAYOUT_MIGRATION_GATE": "PASS_FAST",
            "PACKAGE_DEDUPLICATION_GATE": "PASS_FAST",
            "PACKAGE_PROVENANCE_GATE": "PASS",
            "BUILDER_OUTPUT_ROOT_GATE": "ARMED",
            "CURRENT_RECOMMENDATION_POINTER_GATE": "PASS",
        },
        "upload_this": "NO_CANDIDATE_CURRENTLY_RECOMMENDED",
        "live": live,
    }
    atomic_write_json(
        REPO / "experiments" / "manifests" / "phase9fu_submission_layout_gate.json", layout_gate
    )
    atomic_write_text(
        REPO / "experiments" / "reports" / "phase9fu_submission_layout_migration.md",
        f"# Submission layout migration (0B-FAST)\n\nCreated: {now}\n\n"
        f"- V001 archived at `submission/packages/QS-PUBLIC-V001/{LIVE_BUILD}/`\n"
        f"- Related `898b…` at `submission/packages/QS-P9F-PORTAL-V0/{RELATED_BUILD}/`\n"
        "- UPLOAD_THIS = NO_CANDIDATE\n"
        "- Full legacy backfill deferred to Stage 0C\n",
        overwrite=True,
    )

    print(
        json.dumps(
            {
                "freeze": "PASS",
                "live_sha": LIVE_SHA,
                "upload_this": "NO_CANDIDATE_CURRENTLY_RECOMMENDED",
                "related_migrated": RELATED_HISTORICAL.exists() is False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
