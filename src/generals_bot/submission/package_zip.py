"""Resolve and validate submission package ZIP paths."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


class PackageZipError(ValueError):
    """Raised when a package ZIP path is missing, ambiguous, or not a ZIP file."""


def resolve_package_zip(
    *,
    package_zip: Path | None = None,
    search_dir: Path | None = None,
    pattern: str = "*_packaged.zip",
    max_depth: int = 2,
) -> Path:
    """Return exactly one regular ZIP file path.

    Prefer an explicit ``package_zip``. Otherwise search ``search_dir`` for
    ``pattern`` up to ``max_depth`` and require exactly one match.
    """
    if package_zip is not None:
        path = Path(package_zip)
        if path.is_dir():
            raise PackageZipError(
                f"package path is a directory (Docker often creates this when the "
                f"host ZIP is missing): {path}"
            )
        if not path.exists():
            raise PackageZipError(f"package path does not exist: {path}")
        if not path.is_file():
            raise PackageZipError(f"package path is not a regular file: {path}")
        if not zipfile.is_zipfile(path):
            raise PackageZipError(f"package path is not a valid ZIP: {path}")
        return path.resolve()

    if search_dir is None:
        raise PackageZipError("either package_zip or search_dir is required")

    root = Path(search_dir)
    if not root.is_dir():
        raise PackageZipError(f"search directory does not exist: {root}")

    matches: list[Path] = []
    for depth in range(max_depth + 1):
        if depth == 0:
            matches.extend(p for p in root.glob(pattern) if p.is_file())
        else:
            glob_pat = "/".join(["*"] * depth) + "/" + pattern
            matches.extend(p for p in root.glob(glob_pat) if p.is_file())

    seen: set[Path] = set()
    unique: list[Path] = []
    for p in matches:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(rp)

    if len(unique) == 0:
        raise PackageZipError(f"no package ZIP matching {pattern!r} under {root}")
    if len(unique) > 1:
        listing = "\n".join(str(p) for p in unique)
        raise PackageZipError(f"expected exactly one package ZIP, found {len(unique)}:\n{listing}")

    path = unique[0]
    if not zipfile.is_zipfile(path):
        raise PackageZipError(f"matched path is not a valid ZIP: {path}")
    return path


def package_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def zip_root_entries(path: Path) -> list[str]:
    with zipfile.ZipFile(path, "r") as zf:
        return sorted(zf.namelist())
