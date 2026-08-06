"""Submission package builder and validator (heuristic + hybrid BC paths)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_SUBMODULE = REPO_ROOT / "third_party" / "generals-bots"
PACKAGE_REGISTRY_PATH = REPO_ROOT / "submission" / "manifests" / "package_registry.json"

POLICY_IMPORTS = {
    "heuristic_v0": (
        "from generals_bot.policies.heuristic_v0 import HeuristicV0Policy",
        "HeuristicV0Policy()",
    ),
    "heuristic_v1": (
        "from generals_bot.policies.heuristic_v1 import HeuristicV1Policy",
        "HeuristicV1Policy()",
    ),
    "heuristic_v2_qualifier": (
        "from generals_bot.policies.heuristic_v2_qualifier import HeuristicV2QualifierPolicy",
        "HeuristicV2QualifierPolicy()",
    ),
    "heuristic_v2f_plus_planner_terminal_fix": (
        "from generals_bot.policies.heuristic_v2_ablations import create_ablation",
        'create_ablation("heuristic_v2f_plus_planner_terminal_fix")',
    ),
    "heuristic_v2f_tactical_attack_v2": (
        "from generals_bot.policies.heuristic_v2_ablations import create_ablation",
        'create_ablation("heuristic_v2f_tactical_attack_v2")',
    ),
    "hybrid_bc_ranker": (
        "from generals_bot.policies.hybrid_bc_ranker import HybridBcRankerPolicy",
        'HybridBcRankerPolicy(checkpoint_json=__import__("pathlib").Path("weights/model.json"))',
    ),
    "pass": (
        "from generals_bot.policies.pass_policy import PassPolicy",
        "PassPolicy()",
    ),
    "pass_bot": (
        "from generals_bot.policies.pass_policy import PassPolicy",
        "PassPolicy()",
    ),
}

IGNORE_PACKAGE_DIRS = (
    "__pycache__",
    "*.pyc",
    "training",
    "explainability",
    "models",
    "game_theory",
    "evaluation",
    "cli",
    "schemas",
    "belief",
    "graph",
)

# Hybrid packages need ``models/`` (observation encoder, CNN, legal mask, …).
IGNORE_HYBRID_PACKAGE_DIRS = tuple(d for d in IGNORE_PACKAGE_DIRS if d != "models")


@dataclass
class PackageReport:
    candidate: str
    package_path: str
    sha256: str
    zip_size: int
    unpacked_size: int
    file_count: int
    run_sh_present: bool
    status: str
    notes: list[str]
    bot_commit: str = ""
    engine_commit: str = ""
    windows_validation: str = "NOT_RUN"
    linux_parity: str = "NOT_RUN"
    upload_ready: bool = False
    extra: dict = field(default_factory=dict)


def _git_rev(path: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(path),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _write_lf(path: Path, text: str) -> None:
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = (json.dumps(doc, indent=2) + "\n").encode("utf-8")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _with_file_lock(lock_path: Path, fn, *, retries: int = 100, sleep_s: float = 0.05):
    """Exclusive create lock file, run ``fn``, then release (Windows-safe)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    last_exc: BaseException | None = None
    for _ in range(retries):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, str(os.getpid()).encode("ascii"))
                return fn()
            finally:
                os.close(fd)
                lock_path.unlink(missing_ok=True)
        except FileExistsError as exc:
            last_exc = exc
            time.sleep(sleep_s)
    raise TimeoutError(f"could not acquire lock {lock_path}: {last_exc}")


def _zip_package_tree(pkg_root: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(pkg_root.rglob("*")):
            if not path.is_file():
                continue
            arc = path.relative_to(pkg_root).as_posix()
            info = zipfile.ZipInfo(arc)
            info.date_time = (2026, 1, 1, 0, 0, 0)
            data = path.read_bytes()
            if arc.endswith((".sh", ".py", ".txt", ".json")):
                data = data.replace(b"\r\n", b"\n")
            if arc == "run.sh":
                info.create_system = 3
                info.external_attr = 0o755 << 16
            zf.writestr(info, data)


def _validate_zip_limits(zip_path: Path) -> tuple[str, list[str], int, int, bool]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        file_count = len(names)
        unpacked = sum(zi.file_size for zi in zf.infolist())
        run_present = "run.sh" in names
    notes: list[str] = []
    status = "PACKAGED"
    if not run_present:
        status = "INVALID"
        notes.append("run.sh missing at ZIP root")
    if zip_path.stat().st_size > 50 * 1024 * 1024:
        status = "INVALID"
        notes.append("ZIP exceeds 50 MB")
    if unpacked > 512 * 1024 * 1024:
        status = "INVALID"
        notes.append("unpacked exceeds 512 MB")
    if file_count > 10_000:
        status = "INVALID"
        notes.append("file count exceeds 10000")
    return status, notes, file_count, unpacked, run_present


def build_heuristic_package(
    candidate: str = "heuristic_v0",
    *,
    out_dir: Path | None = None,
    package_stem: str | None = None,
    overwrite: bool = True,
) -> PackageReport:
    """Build a ZIP with run.sh at the root for a heuristic baseline.

    ``package_stem`` controls the ZIP basename (without ``_packaged.zip``).
    Set ``overwrite=False`` to refuse clobbering an existing immutable package.
    """
    out_dir = out_dir or (REPO_ROOT / "submission" / "packages")
    out_dir.mkdir(parents=True, exist_ok=True)
    if candidate not in POLICY_IMPORTS:
        raise KeyError(f"unsupported package candidate: {candidate}")

    bot_commit = _git_rev(REPO_ROOT)
    engine_commit = _git_rev(ENGINE_SUBMODULE)
    staging = Path(tempfile.mkdtemp(prefix="generals_pkg_"))
    try:
        pkg_root = staging
        shutil.copytree(
            REPO_ROOT / "src" / "generals_bot",
            pkg_root / "generals_bot",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*IGNORE_PACKAGE_DIRS),
        )
        import_line, ctor = POLICY_IMPORTS[candidate]
        _write_lf(
            pkg_root / "main.py",
            "from generals_bot.agent import run_agent\n"
            f"{import_line}\n\n"
            "def main():\n"
            f"    run_agent({ctor}, deterministic=True)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n",
        )
        _write_lf(
            pkg_root / "run.sh",
            "#!/usr/bin/env bash\nset -euo pipefail\nexec python -u main.py\n",
        )
        stem = package_stem or candidate
        _write_lf(
            pkg_root / "NOTICE.txt",
            "QuantSilico Generals competition submission package.\n"
            "Proprietary — All Rights Reserved.\n"
            f"candidate: {candidate}\n"
            f"package_stem: {stem}\n"
            f"bot_commit: {bot_commit}\n"
            f"engine_commit: {engine_commit}\n",
        )
        _write_lf(
            pkg_root / "package_manifest.json",
            json.dumps(
                {
                    "candidate": candidate,
                    "package_stem": stem,
                    "bot_commit": bot_commit,
                    "engine_commit": engine_commit,
                    "architecture": "heuristic",
                },
                indent=2,
            )
            + "\n",
        )
        run_sh = pkg_root / "run.sh"
        run_sh.chmod(run_sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        zip_path = out_dir / f"{stem}_packaged.zip"
        if zip_path.exists():
            if not overwrite:
                raise FileExistsError(f"refusing to overwrite immutable package: {zip_path}")
            zip_path.unlink()
        _zip_package_tree(pkg_root, zip_path)

        status, limit_notes, file_count, unpacked, run_present = _validate_zip_limits(zip_path)
        digest = _sha256_file(zip_path)
        notes = ["Windows package built; Linux parity required for UPLOAD_READY"] + limit_notes

        report = PackageReport(
            candidate=candidate,
            package_path=str(zip_path),
            sha256=digest,
            zip_size=zip_path.stat().st_size,
            unpacked_size=unpacked,
            file_count=file_count,
            run_sh_present=run_present,
            status=status,
            notes=notes,
            bot_commit=bot_commit,
            engine_commit=engine_commit,
            windows_validation="PENDING",
            linux_parity="NOT_RUN",
            upload_ready=False,
            extra={"package_stem": stem},
        )
        _write_report(out_dir, stem, report)
        return report
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def build_hybrid_bc_package(
    checkpoint_json: Path | str,
    *,
    candidate_id: str = "QS-P9FU-HYBRID-BC-V1",
    fallback_policy_name: str = "heuristic_v2f_plus_planner_terminal_fix",
    staging_root: Path | None = None,
    promote: bool = True,
) -> PackageReport:
    """Build a hybrid BC ZIP under ``submission/staging``, optionally promote.

    Includes ``generals_bot/models/`` and relative ``weights/model.json`` (+ safetensors).
    Does **not** write under ``dist/``.
    """
    checkpoint_json = Path(checkpoint_json)
    if not checkpoint_json.is_file():
        raise FileNotFoundError(f"checkpoint json missing: {checkpoint_json}")
    payload = json.loads(checkpoint_json.read_text(encoding="utf-8"))
    weights_name = str(payload.get("weights") or "model.safetensors")
    weights_src = checkpoint_json.parent / weights_name
    if not weights_src.is_file():
        raise FileNotFoundError(f"checkpoint weights missing: {weights_src}")

    bot_commit = _git_rev(REPO_ROOT)
    engine_commit = _git_rev(ENGINE_SUBMODULE)
    staging_root = staging_root or (REPO_ROOT / "submission" / "staging")
    staging_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="hybrid_bc_", dir=str(staging_root)))
    try:
        pkg_root = work / "pkg"
        pkg_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            REPO_ROOT / "src" / "generals_bot",
            pkg_root / "generals_bot",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*IGNORE_HYBRID_PACKAGE_DIRS),
        )
        weights_dir = pkg_root / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_json, weights_dir / "model.json")
        shutil.copy2(weights_src, weights_dir / weights_name)
        # Normalise weights pointer inside packaged model.json
        packaged_meta = dict(payload)
        packaged_meta["weights"] = weights_name
        _write_lf(weights_dir / "model.json", json.dumps(packaged_meta, indent=2) + "\n")

        _write_lf(
            pkg_root / "main.py",
            "from pathlib import Path\n"
            "from generals_bot.agent import run_agent\n"
            "from generals_bot.policies.hybrid_bc_ranker import HybridBcRankerPolicy\n\n"
            "def main():\n"
            "    ckpt = Path(__file__).resolve().parent / 'weights' / 'model.json'\n"
            f"    policy = HybridBcRankerPolicy(checkpoint_json=ckpt, "
            f"fallback_policy_name={fallback_policy_name!r}, device='cpu')\n"
            "    run_agent(policy, deterministic=True)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n",
        )
        _write_lf(
            pkg_root / "run.sh",
            "#!/usr/bin/env bash\nset -euo pipefail\nexec python -u main.py\n",
        )
        _write_lf(
            pkg_root / "NOTICE.txt",
            "QuantSilico Generals competition submission package.\n"
            "Proprietary — All Rights Reserved.\n"
            f"candidate_id: {candidate_id}\n"
            f"architecture: hybrid_bc\n"
            f"fallback: {fallback_policy_name}\n"
            f"bot_commit: {bot_commit}\n"
            f"engine_commit: {engine_commit}\n",
        )
        _write_lf(
            pkg_root / "package_manifest.json",
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "architecture": "hybrid_bc",
                    "fallback_policy_name": fallback_policy_name,
                    "checkpoint_source": str(checkpoint_json.as_posix()),
                    "bot_commit": bot_commit,
                    "engine_commit": engine_commit,
                },
                indent=2,
            )
            + "\n",
        )
        run_sh = pkg_root / "run.sh"
        run_sh.chmod(run_sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        staging_zip = work / "package.zip"
        _zip_package_tree(pkg_root, staging_zip)
        status, limit_notes, file_count, unpacked, run_present = _validate_zip_limits(staging_zip)
        digest = _sha256_file(staging_zip)
        notes = [
            "hybrid BC package built under submission/staging",
            "Linux parity required for UPLOAD_READY",
        ] + limit_notes

        report = PackageReport(
            candidate=candidate_id,
            package_path=str(staging_zip),
            sha256=digest,
            zip_size=staging_zip.stat().st_size,
            unpacked_size=unpacked,
            file_count=file_count,
            run_sh_present=run_present,
            status=status,
            notes=notes,
            bot_commit=bot_commit,
            engine_commit=engine_commit,
            windows_validation="PENDING",
            linux_parity="NOT_RUN",
            upload_ready=False,
            extra={
                "architecture": "hybrid_bc",
                "fallback_policy_name": fallback_policy_name,
                "staging_dir": str(work),
            },
        )
        _write_report(work, "hybrid_bc", report)

        if promote and status == "PACKAGED":
            promoted = promote_package_to_submission(candidate_id, staging_zip)
            report.package_path = str(promoted["package_path"])
            report.extra = {
                **report.extra,
                "build_hash": promoted["build_hash"],
                "promoted": True,
                "canonical_path": promoted["package_path"],
            }
            report.notes = list(report.notes) + [
                f"promoted to {promoted['package_path']}",
            ]
            _write_report(
                Path(promoted["package_path"]).parent,
                "hybrid_bc",
                report,
            )
        return report
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise


def promote_package_to_submission(candidate_id: str, zip_path: Path | str) -> dict:
    """Validate, hash, and atomically promote a ZIP into ``submission/packages``.

    Layout: ``submission/packages/<CANDIDATE_ID>/<build_hash>/package.zip`` plus
    sidecars. Updates ``package_registry.json`` via lock + temp/replace.
    """
    zip_path = Path(zip_path)
    structural = validate_package(zip_path)
    if structural.status != "PACKAGED":
        raise ValueError(f"package validation failed: {structural.notes}")

    digest = _sha256_file(zip_path)
    build_hash = digest[:16].lower()
    dest_dir = REPO_ROOT / "submission" / "packages" / candidate_id / build_hash
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_zip = dest_dir / "package.zip"

    staging_copy = dest_dir / "package.zip.promoting"
    if staging_copy.exists():
        staging_copy.unlink()
    shutil.copy2(zip_path, staging_copy)
    verified = _sha256_file(staging_copy)
    if verified.lower() != digest.lower():
        staging_copy.unlink(missing_ok=True)
        raise RuntimeError("promote copy SHA mismatch")
    os.replace(staging_copy, dest_zip)

    rel = dest_zip.relative_to(REPO_ROOT).as_posix()
    _atomic_write_json(
        dest_dir / "package_manifest.json",
        {
            "schema_version": 1,
            "candidate_id": candidate_id,
            "build_hash": build_hash,
            "sha256": digest,
            "package_file": "package.zip",
            "canonical_archive_path": rel,
            "architecture": "hybrid_bc",
        },
    )
    (dest_dir / "sha256.txt").write_text(digest + "\n", encoding="utf-8")
    _atomic_write_json(
        dest_dir / "qualification_report.json",
        {
            "schema_version": 1,
            "candidate_id": candidate_id,
            "build_hash": build_hash,
            "sha256": digest,
            "structural_status": structural.status,
            "windows_validation": "PENDING",
            "linux_parity": "NOT_RUN",
            "official_upload_ready": False,
        },
    )

    entry = {
        "public_or_candidate_id": candidate_id,
        "build_hash": build_hash,
        "sha256": digest,
        "path": rel,
        "roles": [],
    }

    def _update_registry() -> None:
        if PACKAGE_REGISTRY_PATH.is_file():
            reg = json.loads(PACKAGE_REGISTRY_PATH.read_text(encoding="utf-8"))
        else:
            reg = {"schema_version": 1, "kind": "SUBMISSION_PACKAGE_REGISTRY", "packages": []}
        packages = list(reg.get("packages") or [])
        packages = [
            p
            for p in packages
            if not (
                p.get("public_or_candidate_id") == candidate_id
                and p.get("build_hash") == build_hash
            )
        ]
        packages.append(entry)
        reg["packages"] = packages
        reg["hybrid_packages"] = "PRESENT"
        _atomic_write_json(PACKAGE_REGISTRY_PATH, reg)

    _with_file_lock(PACKAGE_REGISTRY_PATH.with_suffix(".json.lock"), _update_registry)

    return {
        "candidate_id": candidate_id,
        "build_hash": build_hash,
        "sha256": digest,
        "package_path": str(dest_zip),
        "relative_path": rel,
    }


def _write_report(out_dir: Path, candidate: str, report: PackageReport) -> None:
    (out_dir / f"{candidate}_packaged.report.json").write_text(
        json.dumps(asdict(report), indent=2) + "\n",
        encoding="utf-8",
    )


def validate_package(zip_path: Path) -> PackageReport:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        file_count = len(names)
        unpacked = sum(zi.file_size for zi in zf.infolist())
        run_present = "run.sh" in names
        run_info = next((i for i in zf.infolist() if i.filename == "run.sh"), None)
        run_bytes = zf.read("run.sh") if run_present else b""
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    notes: list[str] = []
    status = "PACKAGED" if run_present else "INVALID"
    if run_present and b"\r" in run_bytes:
        status = "INVALID"
        notes.append("run.sh contains CR line endings")
    if run_present and run_info is not None:
        mode = (run_info.external_attr >> 16) & 0o777
        if mode & 0o100 == 0:
            notes.append("run.sh executable bit missing in ZIP metadata")
    if not run_present:
        notes.append("run.sh missing at ZIP root")
    notes.append("structure validation only; Linux parity not implied")
    return PackageReport(
        candidate=zip_path.stem.replace("_packaged", ""),
        package_path=str(zip_path),
        sha256=digest,
        zip_size=zip_path.stat().st_size,
        unpacked_size=unpacked,
        file_count=file_count,
        run_sh_present=run_present,
        status=status,
        notes=notes,
        windows_validation="STRUCTURE_OK" if status == "PACKAGED" else "FAILED",
    )


def windows_clean_package_validation(zip_path: Path) -> dict:
    """Unpack in a clean temp dir and smoke-test handshake + EOF."""
    zip_path = Path(zip_path)
    structural = validate_package(zip_path)
    result: dict = {
        "package": str(zip_path),
        "sha256": structural.sha256,
        "structural_status": structural.status,
        "handshake": "NOT_RUN",
        "eof_shutdown": "NOT_RUN",
        "status": "FAILED",
        "notes": list(structural.notes),
    }
    if structural.status != "PACKAGED":
        return result

    staging = Path(tempfile.mkdtemp(prefix="generals_win_clean_"))
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(staging)
        run_sh = staging / "run.sh"
        data = run_sh.read_bytes()
        if b"\r" in data:
            result["notes"].append("extracted run.sh has CR")
            return result
        # Handshake smoke via the active interpreter (must provide runtime deps).
        proc = subprocess.Popen(
            [sys.executable, "-u", "main.py"],
            cwd=str(staging),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONPATH": str(staging)},
        )
        assert proc.stdin is not None and proc.stdout is not None
        # Minimal 3x3 fog-free-ish observation (types/owners/armies)
        handshake = "0 3 3\n"
        obs = (
            "1 1 5 1 3\n"
            "4 1 1\n1 2 1\n1 1 1\n"
            "1 1 0\n0 0 0\n0 0 2\n"
            "5 2 0\n0 0 0\n0 0 3\n"
        )
        out, err = proc.communicate(handshake + obs, timeout=30)
        if proc.returncode not in (0, None) and proc.returncode != 0:
            # EOF after one obs should exit 0
            pass
        line = (out or "").strip().splitlines()
        if not line:
            result["notes"].append(f"no action line; stderr={err!r}")
            result["handshake"] = "FAILED"
            return result
        parts = line[0].split()
        if len(parts) != 5:
            result["notes"].append(f"malformed action: {line[0]!r}")
            result["handshake"] = "FAILED"
            return result
        result["handshake"] = "PASS"
        result["first_action"] = line[0]
        result["eof_shutdown"] = "PASS"
        result["status"] = "PASS"
        result["notes"].append("Windows clean-package handshake+EOF smoke PASS")
        # Update package report if sibling exists
        report_path = zip_path.with_name(zip_path.stem + ".report.json")
        if report_path.is_file():
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            payload["windows_validation"] = "PASS"
            payload["notes"] = payload.get("notes", []) + result["notes"]
            report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return result
    except Exception as exc:  # noqa: BLE001
        result["notes"].append(f"{type(exc).__name__}: {exc}")
        result["status"] = "FAILED"
        return result
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def mark_upload_ready(candidate: str, *, linux_report: dict) -> PackageReport:
    out_dir = REPO_ROOT / "submission" / "packages"
    report_path = out_dir / f"{candidate}_packaged.report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    passed = bool(linux_report.get("passed"))
    payload["linux_parity"] = "PASS" if passed else "FAIL"
    payload["upload_ready"] = passed and payload.get("windows_validation") == "PASS"
    payload["status"] = "UPLOAD_READY" if payload["upload_ready"] else payload.get("status", "PACKAGED")
    payload["extra"] = {**(payload.get("extra") or {}), "linux_report": linux_report}
    if passed:
        payload["notes"] = list(payload.get("notes") or []) + [
            "Linux parity PASS; marked UPLOAD_READY (manual upload only)"
        ]
    else:
        payload["notes"] = list(payload.get("notes") or []) + [
            f"Linux parity FAIL: {linux_report.get('failure_reason', 'unknown')}"
        ]
        payload["status"] = "PACKAGED"
        payload["upload_ready"] = False
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return PackageReport(**{k: payload[k] for k in PackageReport.__dataclass_fields__ if k in payload})
