"""Fixture: emits an extra action line each turn."""

from __future__ import annotations

import sys


def main() -> None:
    handshake = sys.stdin.readline()
    if not handshake:
        return
    _pid, h, _w = (int(x) for x in handshake.split())
    while True:
        first_line = sys.stdin.readline()
        if not first_line:
            return
        for _ in range(3 * h):
            sys.stdin.readline()
        sys.stdout.write("1 0 0 0 0\n")
        sys.stdout.write("1 0 0 0 0\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
