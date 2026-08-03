"""Legal-random baseline entrypoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from generals_bot.agent import run_agent
from generals_bot.policies.random_policy import RandomPolicy


def main() -> None:
    seed = int(os.environ.get("GENERALS_BOT_SEED", "0"))
    run_agent(RandomPolicy(seed=seed), deterministic=False)


if __name__ == "__main__":
    main()
