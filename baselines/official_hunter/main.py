"""Official hunter baseline entrypoint (EVAL_ONLY, calibration fixture).

Serves the pinned engine's HunterAgent (a general-targeting attacker) through
the official protocol. Used by EVAL_WIN_CONVERSION_CALIBRATION_V1 as the
forced-win probe: a known general-hunter MUST beat an always-pass opponent
inside the 1200-turn horizon, otherwise engine/evaluator semantics require
investigation before draw-heavy candidate results are read as policy weakness
(audit amendment sections 19/20).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
_REPO = Path(__file__).resolve().parents[2]
for entry in (_SRC, _REPO):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from generals_bot.agent import run_agent  # noqa: E402
from generals_bot.policies.official_hunter import OfficialHunterPolicy  # noqa: E402


def main() -> None:
    run_agent(OfficialHunterPolicy(), deterministic=True)


if __name__ == "__main__":
    main()
