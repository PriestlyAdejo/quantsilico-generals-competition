"""S5-T2-K1-S2 serving candidate (EVAL_ONLY) for the STAGE5 T2 gameplay arbiter.

Serves the registered terminal checkpoint (raw weights) through the official
stdio protocol via s5_t2_serving_common (k1 temporal history adapter).
PPO_SEMANTICS: EVAL_ONLY.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BASELINES = Path(__file__).resolve().parents[1]
if str(_BASELINES) not in sys.path:
    sys.path.insert(0, str(_BASELINES))

from s5_t2_serving_common import serve  # noqa: E402

if __name__ == "__main__":
    serve("S5-T2-K1-S2")
