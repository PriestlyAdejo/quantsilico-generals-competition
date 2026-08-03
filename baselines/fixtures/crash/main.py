"""Fixture: crashes immediately after handshake."""

from __future__ import annotations

import sys


def main() -> None:
    handshake = sys.stdin.readline()
    if not handshake:
        return
    raise SystemExit(1)


if __name__ == "__main__":
    main()
