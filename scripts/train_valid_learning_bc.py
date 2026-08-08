#!/usr/bin/env python3
"""CLI for the competition-native BC warm-start gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from train.competition_native_jax.bc_warmstart_jax import train_warmstart


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--wall-minutes", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--evaluation-interval", type=int, default=50)
    parser.add_argument(
        "--trainable-scope",
        choices=("full_policy", "policy_heads", "last_layer_heads"),
        default="full_policy",
    )
    args = parser.parse_args()
    identity = hashlib.sha256(
        json.dumps(
            {
                "dataset": str(args.dataset.resolve()),
                "parent": str(args.parent.resolve()),
                "steps": args.steps,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "evaluation_interval": args.evaluation_interval,
                "trainable_scope": args.trainable_scope,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    output = args.runtime / "bc" / f"warmstart_{identity[:16]}"
    report = train_warmstart(
        dataset_path=args.dataset,
        parent_checkpoint=args.parent,
        output=output,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        wall_minutes=args.wall_minutes,
        seed=args.seed,
        evaluation_interval=args.evaluation_interval,
        trainable_scope=args.trainable_scope,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "BC_NUMERIC_GATE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
