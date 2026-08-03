"""Pass-bot baseline entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without editable install when PYTHONPATH is unset.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from generals_bot.agent import run_agent
from generals_bot.policies.pass_policy import PassPolicy


def main() -> None:
    run_agent(PassPolicy(), deterministic=True)


if __name__ == "__main__":
    main()
