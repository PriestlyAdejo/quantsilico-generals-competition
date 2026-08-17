"""OBS_V2_R1 arm OBS-V2-R1-S1 protocol entrypoint (EVAL_ONLY)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obs_v2_serving_common import serve  # noqa: E402

if __name__ == "__main__":
    serve("OBS-V2-R1-S1")
