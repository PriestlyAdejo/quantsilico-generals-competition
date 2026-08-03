"""Heuristic v1 baseline entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from generals_bot.agent import run_agent
from generals_bot.policies.heuristic_v1 import HeuristicV1Policy


def main() -> None:
    run_agent(HeuristicV1Policy(), deterministic=True)


if __name__ == "__main__":
    main()
