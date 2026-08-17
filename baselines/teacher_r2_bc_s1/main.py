"""STAGE5_TEACHER_R2 BC candidate protocol entrypoint (EVAL_ONLY)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teacher_r2_serving_common import serve  # noqa: E402

if __name__ == "__main__":
    serve()
