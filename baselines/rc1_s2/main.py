"""RC_R1_BRIDGE arm RC-R1-BRIDGE-S2 protocol entrypoint (EVAL_ONLY)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rc1_serving_common import serve  # noqa: E402

if __name__ == "__main__":
    serve("RC-R1-BRIDGE-S2")
