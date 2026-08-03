"""Fixture: responds late every turn."""

from __future__ import annotations

import sys
import time


def main() -> None:
    handshake = sys.stdin.readline()
    if not handshake:
        return
    _pid, h, _w = (int(x) for x in handshake.split())
    first = True
    while True:
        first_line = sys.stdin.readline()
        if not first_line:
            return
        for _ in range(3 * h):
            sys.stdin.readline()
        time.sleep(0.05 if first else 0.35)
        first = False
        sys.stdout.write("1 0 0 0 0\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
