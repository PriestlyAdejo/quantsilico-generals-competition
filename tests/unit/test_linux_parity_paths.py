"""Regression tests for Linux parity package path resolution."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from generals_bot.submission.package_zip import PackageZipError, resolve_package_zip


def _write_minimal_zip(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("run.sh", "#!/usr/bin/env bash\necho ok\n")
        zf.writestr("main.py", "print('ok')\n")
    return path


def test_resolve_rejects_directory(tmp_path: Path) -> None:
    d = tmp_path / "package.zip"
    d.mkdir()
    with pytest.raises(PackageZipError, match="directory"):
        resolve_package_zip(package_zip=d)


def test_resolve_rejects_absent(tmp_path: Path) -> None:
    with pytest.raises(PackageZipError, match="does not exist"):
        resolve_package_zip(package_zip=tmp_path / "missing_packaged.zip")


def test_resolve_rejects_multiple(tmp_path: Path) -> None:
    _write_minimal_zip(tmp_path / "a_packaged.zip")
    _write_minimal_zip(tmp_path / "b_packaged.zip")
    with pytest.raises(PackageZipError, match="exactly one"):
        resolve_package_zip(search_dir=tmp_path)


def test_resolve_accepts_single_zip(tmp_path: Path) -> None:
    z = _write_minimal_zip(tmp_path / "heuristic_v1_packaged.zip")
    got = resolve_package_zip(package_zip=z)
    assert got == z.resolve()
    got2 = resolve_package_zip(search_dir=tmp_path)
    assert got2 == z.resolve()


def test_run_parity_rejects_directory(tmp_path: Path) -> None:
    # Import via path load to avoid requiring scripts as a package.
    import importlib.util

    mod_path = Path(__file__).resolve().parents[2] / "scripts" / "parity" / "linux_package_parity.py"
    spec = importlib.util.spec_from_file_location("linux_package_parity", mod_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    d = tmp_path / "package.zip"
    d.mkdir()
    report_out = tmp_path / "report.json"
    report = mod.run_parity(d, report_out)
    assert report["passed"] is False
    assert "directory" in report["failure_reason"].lower()
    assert report_out.is_file()
