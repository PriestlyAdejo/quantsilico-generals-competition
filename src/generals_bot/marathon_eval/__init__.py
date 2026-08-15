"""Marathon paired evaluator (EXECUTION_PLAN Stage 2, programme.yaml evaluation).

Canonical paired evaluation: same map and seed per pair, candidate plays both
seats, sequentially valid anytime inference over paired score differences,
promotion policy from configs/marathon/programme.yaml.
"""

from generals_bot.marathon_eval.confidence_sequence import (
    AnytimeBoundedCS,
    backtransform_interval,
)
from generals_bot.marathon_eval.pairing import (
    PairSchedule,
    pair_schedule,
    pair_score_from_game_scores,
)
from generals_bot.marathon_eval.promotion import PromotionDecision, decide_promotion
from generals_bot.marathon_eval.store import PairedEvalStore, PairResult

__all__ = [
    "AnytimeBoundedCS",
    "backtransform_interval",
    "PairSchedule",
    "pair_schedule",
    "pair_score_from_game_scores",
    "PromotionDecision",
    "decide_promotion",
    "PairResult",
    "PairedEvalStore",
]
