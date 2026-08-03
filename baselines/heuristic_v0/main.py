"""Heuristic v0 baseline entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from generals_bot.agent import run_agent
from generals_bot.policies.heuristic_v0 import HeuristicV0Policy


def main() -> None:
    run_agent(HeuristicV0Policy(), deterministic=True)


if __name__ == "__main__":
    main()
