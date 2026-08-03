"""Fixture: well-formed but illegal move every turn."""

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
        # Move from (0,0) off the board upward — illegal but well-formed.
        sys.stdout.write("0 0 0 0 0\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
