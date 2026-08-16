"""SCALE-A0-8M-S1 serving candidate (EVAL_ONLY) for the STAGE5 SCALE-R1 gameplay arbiter.

Serves the registered terminal checkpoint (raw weights) through the official
stdio protocol via scale_r1_serving_common (canonical 8-plane policy).
PPO_SEMANTICS: EVAL_ONLY.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BASELINES = Path(__file__).resolve().parents[1]
if str(_BASELINES) not in sys.path:
    sys.path.insert(0, str(_BASELINES))

from scale_r1_serving_common import serve  # noqa: E402

if __name__ == "__main__":
    serve("SCALE-A0-8M-S1")
