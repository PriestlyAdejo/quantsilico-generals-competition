"""RWB1-A0-CONTROL-S1 serving candidate (EVAL_ONLY) for the REWARD-BRIDGE-R1 gameplay arbiter.

Serves the registered terminal checkpoint (raw weights) through the official
stdio protocol via rwb1_serving_common. PPO_SEMANTICS: EVAL_ONLY.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BASELINES = Path(__file__).resolve().parents[1]
if str(_BASELINES) not in sys.path:
    sys.path.insert(0, str(_BASELINES))

from rwb1_serving_common import serve  # noqa: E402

if __name__ == "__main__":
    serve("RWB1-A0-CONTROL-S1")
