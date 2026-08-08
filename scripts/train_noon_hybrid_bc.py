#!/usr/bin/env python3
"""Train the bounded Hybrid BC noon-rescue candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from train.competition_native_jax.deadline_rescue_bc_jax import train_rescue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dataset", type=Path, required=True)
    parser.add_argument("--dagger-dataset", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    identity = hashlib.sha256(
        json.dumps(vars(args), default=str, sort_keys=True).encode()
    ).hexdigest()
    output = args.runtime / "hybrid_bc" / f"rescue_{identity[:16]}"
    report = train_rescue(
        original_dataset=args.original_dataset,
        dagger_dataset=args.dagger_dataset,
        parent_checkpoint=args.parent,
        output=output,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
