"""SCALE-B1-16M-S1 serving candidate (EVAL_ONLY) for the STAGE5 SCALE-R2 arbiter.

Canonical 8-plane path via scale_r1_serving_common; checkpoint root
overridable via SCALE_CKPT_DIR.
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
    serve("SCALE-B1-16M-S1")
