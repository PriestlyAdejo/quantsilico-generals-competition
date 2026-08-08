#!/usr/bin/env python3
"""CLI for mandatory post-BC value-head recalibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from train.competition_native_jax.value_calibration_jax import calibrate_value_head


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-len", type=int, default=1_200)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()
    identity = hashlib.sha256(
        json.dumps(vars(args), default=str, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output = args.runtime / "value_calibration" / identity[:16]
    report = calibrate_value_head(
        warmstart_checkpoint=args.warmstart,
        output=output,
        num_envs=args.num_envs,
        rollout_len=args.rollout_len,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "POST_BC_VALUE_CALIBRATION_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
