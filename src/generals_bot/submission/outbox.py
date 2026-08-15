"""Deterministic submission outbox allocation (Stage 4B packaging lane).

Canonical outbox path: ``submission/outbox/qs-<candidate>-vNNN-YYYY-MM-DD.zip``
with matching manifest and SHA-256 sidecar. Guarantees:

- deterministic naming (zero-padded 3-digit version, fixed date string);
- atomic allocation (temp copy in the same directory, then os.replace);
- collision refusal (existing target never overwritten);
- staging cleanup on any failure;
- registry-backed identity (links package_registry.json when the source
  build is a promoted canonical package).

Portal upload is OUT of scope: allocation never uploads anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTBOX_ROOT = REPO_ROOT / "submission" / "outbox"
PACKAGE_REGISTRY_PATH = REPO_ROOT / "submission" / "manifests" / "package_registry.json"

_NAME_RE_TEMPLATE = r"qs-{candidate}-v(?P<version>\d{{3}})-(?P<date>\d{{4}}-\d{{2}}-\d{{2}})\.zip"


class OutboxCollisionError(RuntimeError):
    """The target outbox slot already exists; immutable bytes win."""


@dataclass
class OutboxAllocation:
    candidate_id: str
    version: int
    date: str
    zip_path: Path
    sha256: str
    size: int
    manifest_path: Path
    sidecar_path: Path
    registry_link: dict | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _name_regex(candidate_id: str) -> re.Pattern[str]:
    return re.compile(_NAME_RE_TEMPLATE.format(candidate=re.escape(candidate_id)))


def next_version(candidate_id: str, *, root: Path = DEFAULT_OUTBOX_ROOT) -> int:
    """Next free 3-digit version for this candidate across ALL dates."""
    if not root.is_dir():
        return 1
    pattern = _name_regex(candidate_id)
    highest = 0
    for entry in root.iterdir():
        match = pattern.fullmatch(entry.name)
        if match:
            highest = max(highest, int(match.group("version")))
    return highest + 1


def find_registry_link(candidate_id: str, sha256: str) -> dict | None:
    """Registry-backed identity: match candidate/build hash in package registry."""
    try:
        registry = json.loads(PACKAGE_REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for package in registry.get("packages", []):
        if (
            package.get("public_or_candidate_id") == candidate_id
            and package.get("sha256") == sha256
        ):
            return {
                "candidate_id": candidate_id,
                "build_hash": package.get("build_hash"),
                "canonical_archive_path": package.get("path"),
                "roles": package.get("roles", []),
            }
    return None


def allocate_outbox(
    candidate_id: str,
    source_zip: Path | str,
    *,
    date: str | None = None,
    root: Path | None = None,
    provenance: dict | None = None,
) -> OutboxAllocation:
    """Atomically allocate the next outbox slot for ``candidate_id``.

    Raises OutboxCollisionError if the slot is already occupied and
    ValueError for malformed inputs. Leaves no partial files on failure.
    """
    source_zip = Path(source_zip)
    root = Path(root) if root else DEFAULT_OUTBOX_ROOT
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", candidate_id):
        raise ValueError(f"malformed candidate id: {candidate_id!r}")
    if not source_zip.is_file():
        raise FileNotFoundError(f"source zip missing: {source_zip}")
    date_str = date or time.strftime("%Y-%m-%d", time.gmtime())
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_str):
        raise ValueError(f"malformed date: {date_str!r}")

    digest = _sha256_file(source_zip)
    size = source_zip.stat().st_size
    version = next_version(candidate_id, root=root)
    name = f"qs-{candidate_id}-v{version:03d}-{date_str}.zip"
    target = root / name
    if target.exists():
        raise OutboxCollisionError(f"outbox slot already occupied: {target}")

    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / f".{name}.lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    tmp = root / f".{name}.tmp"
    try:
        os.write(fd, str(os.getpid()).encode("ascii"))
        shutil.copy2(source_zip, tmp)
        if _sha256_file(tmp) != digest:
            raise RuntimeError("outbox staging copy SHA mismatch")
        os.replace(tmp, target)  # atomic allocation
        registry_link = find_registry_link(candidate_id, digest)
        manifest = {
            "schema_version": 1,
            "kind": "SUBMISSION_OUTBOX_MANIFEST",
            "candidate_id": candidate_id,
            "version": version,
            "date": date_str,
            "package_file": name,
            "sha256": digest,
            "size_bytes": size,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": str(source_zip.as_posix()),
            "registry_link": registry_link,
            "provenance": provenance or {},
            "upload_policy": "MANUAL_UPLOAD_ONLY_OUTSIDE_AUTOMATION",
        }
        manifest_path = root / f"{name}.manifest.json"
        sidecar_path = root / f"{name}.sha256"
        manifest_tmp = root / f".{name}.manifest.json.tmp"
        manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(manifest_tmp, manifest_path)
        sidecar_path.write_text(f"{digest}  {name}\n", encoding="utf-8")
        return OutboxAllocation(
            candidate_id=candidate_id,
            version=version,
            date=date_str,
            zip_path=target,
            sha256=digest,
            size=size,
            manifest_path=manifest_path,
            sidecar_path=sidecar_path,
            registry_link=registry_link,
        )
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        os.close(fd)
        lock_path.unlink(missing_ok=True)
