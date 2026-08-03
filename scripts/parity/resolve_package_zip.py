"""CLI wrapper: resolve and validate a submission package ZIP for Linux parity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from generals_bot.submission.package_zip import (
    PackageZipError,
    package_sha256,
    resolve_package_zip,
    zip_root_entries,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve a submission package ZIP for parity")
    parser.add_argument("--package-zip", type=Path, default=None)
    parser.add_argument("--search-dir", type=Path, default=None)
    parser.add_argument("--pattern", default="*_packaged.zip")
    parser.add_argument("--print-sha256", action="store_true")
    parser.add_argument("--print-root", action="store_true")
    args = parser.parse_args(argv)

    try:
        path = resolve_package_zip(
            package_zip=args.package_zip,
            search_dir=args.search_dir,
            pattern=args.pattern,
        )
    except PackageZipError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(path)
    if args.print_sha256:
        print(f"sha256={package_sha256(path)}")
    if args.print_root:
        names = zip_root_entries(path)
        print("root_entries=" + ",".join(names[:20]))
        if "run.sh" not in names:
            print("ERROR: run.sh missing at ZIP root", file=sys.stderr)
            return 1
        print("run_sh_at_root=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
