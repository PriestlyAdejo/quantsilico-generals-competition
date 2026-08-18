"""Run a marathon paired evaluation and emit a promotion decision.

Plays candidate (and optionally incumbent) against one or more protocol
agents over canonical seat-swapped pairs, streams paired differences through
the anytime-valid confidence sequence, and applies the programme.yaml
promotion margins.  Results are atomic and resumable per run directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from generals_bot.marathon_eval.confidence_sequence import AnytimeBoundedCS  # noqa: E402
from generals_bot.marathon_eval.pairing import pair_schedule  # noqa: E402
from generals_bot.marathon_eval.promotion import (  # noqa: E402
    PromotionDecision,
    decide_promotion,
)
from generals_bot.marathon_eval.runner import run_evaluation  # noqa: E402
from generals_bot.marathon_eval.store import PairedEvalStore  # noqa: E402


def load_programme(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def evaluate_side(
    *,
    run_dir: Path,
    agent_id: str,
    agent_main: Path,
    opponents: list[tuple[str, Path]],
    pairs_per_opponent: int,
    eval_namespace: str,
    mode: str,
    max_turns: int | None,
) -> dict[str, float]:
    """Evaluate one agent against all opponents; return pair_id -> score."""
    store_dir = run_dir / agent_id
    schedules = []
    for opponent_id, _ in opponents:
        schedules.extend(
            pair_schedule(
                eval_namespace=eval_namespace,
                opponent_id=opponent_id,
                pair_count=pairs_per_opponent,
            )
        )
    scores: dict[str, float] = {}
    for opponent_id, opponent_main in opponents:
        opponent_schedules = [s for s in schedules if s.opponent_id == opponent_id]
        results = run_evaluation(
            run_dir=store_dir,
            schedules=opponent_schedules,
            candidate_main=agent_main,
            opponent_main=opponent_main,
            candidate_id=agent_id,
            mode=mode,
            max_turns=max_turns,
            on_pair=lambda pair, seconds: print(
                f"pair {pair.pair_id}: score={pair.pair_score} ({seconds:.1f}s)",
                flush=True,
            ),
        )
        scores.update({pair.pair_id: pair.pair_score for pair in results})
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--candidate-main", type=Path, required=True)
    parser.add_argument(
        "--opponent",
        action="append",
        required=True,
        help="id=path to opponent main.py; repeat for multiple opponents",
    )
    parser.add_argument("--incumbent-id", default=None)
    parser.add_argument("--incumbent-main", type=Path, default=None)
    parser.add_argument("--pairs-per-opponent", type=int, default=16)
    parser.add_argument(
        "--run-dir", type=Path, default=REPO / "experiments/marathon/paired_eval_runs/dev"
    )
    parser.add_argument("--eval-namespace", default="marathon-eval-v1")
    parser.add_argument("--mode", default="competition")
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument(
        "--programme", type=Path, default=REPO / "configs/marathon/programme.yaml"
    )
    args = parser.parse_args()

    programme = load_programme(args.programme)
    promotion_cfg = programme["promotion"]
    evaluation_cfg = programme["evaluation"]
    confidence = float(evaluation_cfg.get("confidence", 0.95))
    opponents = []
    for spec in args.opponent:
        opponent_id, _, opponent_path = spec.partition("=")
        opponents.append((opponent_id, Path(opponent_path)))

    started = time.perf_counter()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    candidate_scores = evaluate_side(
        run_dir=args.run_dir,
        agent_id=args.candidate_id,
        agent_main=args.candidate_main,
        opponents=opponents,
        pairs_per_opponent=args.pairs_per_opponent,
        eval_namespace=args.eval_namespace,
        mode=args.mode,
        max_turns=args.max_turns,
    )
    incumbent_scores: dict[str, float] = {}
    if args.incumbent_id and args.incumbent_main:
        incumbent_scores = evaluate_side(
            run_dir=args.run_dir,
            agent_id=args.incumbent_id,
            agent_main=args.incumbent_main,
            opponents=opponents,
            pairs_per_opponent=args.pairs_per_opponent,
            eval_namespace=args.eval_namespace,
            mode=args.mode,
            max_turns=args.max_turns,
        )

    baseline_scores = incumbent_scores or {pair_id: 0.0 for pair_id in candidate_scores}
    differences = [
        candidate_scores[pair_id] - baseline_scores.get(pair_id, 0.0)
        for pair_id in sorted(candidate_scores)
    ]
    cs = AnytimeBoundedCS(alpha=1.0 - confidence)
    interval = cs.interval_on_difference(count=len(differences), difference_total=sum(differences))
    if interval is None:
        print("no pairs completed; nothing to decide", file=sys.stderr)
        return 1
    lower, upper = interval
    store = PairedEvalStore(args.run_dir / args.candidate_id)
    has_incumbent = bool(args.incumbent_id and args.incumbent_main)
    decision = decide_promotion(
        lower_bound=lower,
        practical_margin=float(promotion_cfg["practical_margin"]),
        robustness_lower=lower,
        robustness_noninferiority_margin=float(
            promotion_cfg["robustness_noninferiority_margin"]
        ),
        worst_matchup_improvement=None,  # needs incumbent matchup baselines
        worst_matchup_threshold=float(promotion_cfg["worst_matchup_improvement"]),
        integrity_latency_fault_gates_pass=store.matchup_metrics()["PAIR_COUNT"] > 0,
    )
    if not has_incumbent and decision.promoted:
        # Without an incumbent the differences are raw pair scores against a
        # zero baseline, so a CS lower above the margin measures absolute
        # score, not strength: never promotion evidence (EV-0075 guard).
        decision = PromotionDecision(
            promoted=False,
            pathway="NO_PROMOTION",
            reason=(
                "NO_INCUMBENT_BASELINE: raw-score CS is not strength "
                f"evidence (un-guarded lower {lower:.5f})"
            ),
            lower_bound=lower,
            practical_margin=decision.practical_margin,
        )
    summary = {
        "kind": "MARATHON_PAIRED_EVAL_SUMMARY",
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidate_id": args.candidate_id,
        "incumbent_id": args.incumbent_id,
        "eval_namespace": args.eval_namespace,
        "pairs_completed": len(differences),
        "mean_difference": sum(differences) / len(differences),
        "confidence_sequence": {
            "method": "ANYTIME_VALID_BOUNDED_MIXTURE_CONFIDENCE_SEQUENCE",
            "configured_method": evaluation_cfg.get("sequential_method"),
            "confidence": confidence,
            "lower": lower,
            "upper": upper,
        },
        "matchup_metrics": store.matchup_metrics(),
        "promotion": {
            "promoted": decision.promoted,
            "pathway": decision.pathway,
            "reason": decision.reason,
            "practical_margin": decision.practical_margin,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    store.write_summary(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
