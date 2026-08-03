"""Fixture: never responds (hangs until killed / timeout)."""

from __future__ import annotations

import sys
import time


def main() -> None:
    handshake = sys.stdin.readline()
    if not handshake:
        return
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        # Consume grids but never reply.
        # Handshake told us H W — parse from handshake.
        parts = handshake.split()
        h = int(parts[1])
        for _ in range(3 * h):
            sys.stdin.readline()
        time.sleep(3600)


if __name__ == "__main__":
    main()
