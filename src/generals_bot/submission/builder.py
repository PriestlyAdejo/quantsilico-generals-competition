"""Submission package builder and validator (heuristic path)."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

POLICY_IMPORTS = {
    "heuristic_v0": (
        "from generals_bot.policies.heuristic_v0 import HeuristicV0Policy",
        "HeuristicV0Policy()",
    ),
    "heuristic_v1": (
        "from generals_bot.policies.heuristic_v1 import HeuristicV1Policy",
        "HeuristicV1Policy()",
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


def _write_lf(path: Path, text: str) -> None:
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def build_heuristic_package(
    candidate: str = "heuristic_v0",
    *,
    out_dir: Path | None = None,
) -> PackageReport:
    """Build a ZIP with run.sh at the root for a heuristic baseline."""
    out_dir = out_dir or (REPO_ROOT / "submission" / "packages")
    out_dir.mkdir(parents=True, exist_ok=True)
    if candidate not in POLICY_IMPORTS:
        raise KeyError(f"unsupported package candidate: {candidate}")

    staging = Path(tempfile.mkdtemp(prefix="generals_pkg_"))
    try:
        pkg_root = staging
        shutil.copytree(
            REPO_ROOT / "src" / "generals_bot",
            pkg_root / "generals_bot",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.pyc",
                "training",
                "explainability",
                "models",
            ),
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
        run_sh = pkg_root / "run.sh"
        run_sh.chmod(run_sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        zip_path = out_dir / f"{candidate}_packaged.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(pkg_root.rglob("*")):
                if not path.is_file():
                    continue
                arc = path.relative_to(pkg_root).as_posix()
                info = zipfile.ZipInfo(arc)
                info.date_time = (2026, 1, 1, 0, 0, 0)
                data = path.read_bytes()
                if arc == "run.sh":
                    info.create_system = 3
                    info.external_attr = 0o755 << 16
                    data = data.replace(b"\r\n", b"\n")
                zf.writestr(info, data)

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            file_count = len(names)
            unpacked = sum(zi.file_size for zi in zf.infolist())
            run_present = "run.sh" in names

        digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        notes = ["UPLOAD_READY requires Linux parity gate"]
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
        )
        (out_dir / f"{candidate}_packaged.report.json").write_text(
            json.dumps(asdict(report), indent=2) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def validate_package(zip_path: Path) -> PackageReport:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        file_count = len(names)
        unpacked = sum(zi.file_size for zi in zf.infolist())
        run_present = "run.sh" in names
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    notes = ["structure validation only; Linux parity not run"]
    status = "PACKAGED" if run_present else "INVALID"
    return PackageReport(
        candidate=zip_path.stem,
        package_path=str(zip_path),
        sha256=digest,
        zip_size=zip_path.stat().st_size,
        unpacked_size=unpacked,
        file_count=file_count,
        run_sh_present=run_present,
        status=status,
        notes=notes,
    )
