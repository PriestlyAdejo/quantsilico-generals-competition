"""Stage 2 paired evaluator unit tests (pairing, CS validity, promotion, store)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from generals_bot.marathon_eval.confidence_sequence import (
    AnytimeBoundedCS,
    mixture_log_martingale,
)
from generals_bot.marathon_eval.pairing import (
    canonical_map_seed,
    game_score,
    pair_schedule,
    pair_score_from_game_scores,
)
from generals_bot.marathon_eval.promotion import decide_promotion
from generals_bot.marathon_eval.store import (
    GameRecord,
    PairedEvalStore,
    PairResult,
    replay_identity,
)


def test_map_seeds_are_deterministic_and_disjoint() -> None:
    seed_a = canonical_map_seed("marathon-eval-v1", "opponent-x", 7)
    seed_b = canonical_map_seed("marathon-eval-v1", "opponent-x", 7)
    assert seed_a == seed_b
    assert canonical_map_seed("marathon-eval-v1", "opponent-x", 8) != seed_a
    assert canonical_map_seed("training", "opponent-x", 7) != seed_a
    assert canonical_map_seed("screening", "opponent-x", 7) != seed_a
    assert 0 <= seed_a < 2**31 - 1


def test_pair_schedule_seat_swap() -> None:
    schedules = pair_schedule(
        eval_namespace="marathon-eval-v1", opponent_id="legal_random", pair_count=3
    )
    assert len(schedules) == 3
    assert {s.map_seed for s in schedules} == {s.map_seed for s in schedules}
    assert len({s.map_seed for s in schedules}) == 3
    for schedule in schedules:
        assert schedule.game_seats == (("A", 0), ("B", 1))
        assert schedule.game_count == 2
    resumed = pair_schedule(
        eval_namespace="marathon-eval-v1",
        opponent_id="legal_random",
        pair_count=2,
        start_index=3,
    )
    assert [s.pair_id for s in resumed] == ["legal_random#3", "legal_random#4"]


def test_pair_and_game_scores() -> None:
    assert pair_score_from_game_scores(1.0, 0.0) == 0.5
    assert pair_score_from_game_scores(1.0, 1.0) == 1.0
    assert pair_score_from_game_scores(0.5, 0.0) == 0.25
    assert game_score("WIN") == 1.0
    assert game_score("draw") == 0.5
    assert game_score("LOSS") == 0.0
    with pytest.raises(ValueError):
        game_score("VICTORY")


def test_promotion_normal_boundary_and_robustness() -> None:
    normal = decide_promotion(lower_bound=0.02, practical_margin=0.01)
    assert normal.promoted and normal.pathway == "NORMAL"

    boundary = decide_promotion(lower_bound=0.01, practical_margin=0.01)
    assert not boundary.promoted

    robust = decide_promotion(
        lower_bound=0.0,
        robustness_lower=-0.004,
        worst_matchup_improvement=0.06,
    )
    assert robust.promoted and robust.pathway == "ROBUSTNESS"

    blocked = decide_promotion(
        lower_bound=0.5, integrity_latency_fault_gates_pass=False
    )
    assert not blocked.promoted and blocked.pathway == "NO_PROMOTION"

    not_robust = decide_promotion(
        lower_bound=0.0, robustness_lower=-0.02, worst_matchup_improvement=0.2
    )
    assert not not_robust.promoted


def _simulate_null_pair_differences(rng: random.Random, pairs: int) -> list[float]:
    """Symmetric null model: independent game outcomes, no pairing advantage."""
    differences = []
    for _ in range(pairs):
        seat_a = 1.0 if rng.random() < 0.5 else 0.0
        seat_b = 1.0 if rng.random() < 0.5 else 0.0
        differences.append(seat_a + seat_b - 1.0)
    return differences


@pytest.mark.parametrize("peek", [False, True], ids=["fixed-time", "optional-stopping"])
def test_confidence_sequence_coverage_under_null(peek: bool) -> None:
    """Under the null (true difference 0), the CS excludes 0 with prob <= alpha,

    including when the interval is inspected after every pair.
    """
    cs = AnytimeBoundedCS(alpha=0.05)
    simulations = 300
    pairs = 40
    rng = random.Random(20260815)
    violations = 0
    for _ in range(simulations):
        differences = _simulate_null_pair_differences(rng, pairs)
        total = 0.0
        violated = False
        for count, difference in enumerate(differences, start=1):
            total += difference
            interval = cs.interval_on_difference(count=count, difference_total=total)
            assert interval is not None
            lower, upper = interval
            excludes_zero = lower > 0.0 or upper < 0.0
            if peek and excludes_zero:
                violated = True
                break
        if not peek:
            lower, upper = cs.interval_on_difference(
                count=pairs, difference_total=total
            )
            assert (lower, upper) is not None
            violated = lower > 0.0 or upper < 0.0
        if violated:
            violations += 1
    # E[violations] ~= 15 at alpha=0.05; allow generous slack, catch gross bugs.
    assert violations <= 40, f"{violations}/{simulations} coverage violations"


def test_confidence_sequence_rejects_large_true_difference() -> None:
    cs = AnytimeBoundedCS(alpha=0.05)
    total = 0.0
    interval = None
    for count in range(1, 501):
        total += 0.6  # large positive paired difference
        interval = cs.interval_on_difference(count=count, difference_total=total)
        assert interval is not None
        if interval[0] > 0.0:
            break
    assert interval is not None and interval[0] > 0.0


def test_confidence_sequence_validation() -> None:
    with pytest.raises(ValueError):
        AnytimeBoundedCS(alpha=0.0)
    with pytest.raises(ValueError):
        AnytimeBoundedCS(rho_squared=-1.0)
    assert AnytimeBoundedCS().update(count=0, total=0.0) is None


def test_confidence_sequence_inverts_mixture_test_exactly() -> None:
    """CS endpoints must coincide with the inverted mixture supermartingale.

    At each endpoint mu the log-martingale equals log(1/alpha); inside the
    interval it stays below the rejection threshold.
    """
    import math

    cs = AnytimeBoundedCS(alpha=0.05)
    count, difference_total = 25, 6.0  # shifted: total = (6 + 25) / 2
    shifted_total = (difference_total + count) / 2.0
    lower_shifted, upper_shifted = cs.update(count=count, total=shifted_total)
    threshold = math.log(1.0 / cs.alpha)
    for endpoint in (lower_shifted, upper_shifted):
        value = mixture_log_martingale(
            count=count,
            total=shifted_total,
            mu=endpoint,
            sigma_squared=cs.sigma_squared,
            rho_squared=cs.rho_squared,
        )
        assert value == pytest.approx(threshold, rel=1e-9)
    mean = shifted_total / count
    inside = mixture_log_martingale(
        count=count,
        total=shifted_total,
        mu=mean,
        sigma_squared=cs.sigma_squared,
        rho_squared=cs.rho_squared,
    )
    assert inside < threshold


def _game(pair_id: str, seat: int, outcome: str, opponent: str = "opp") -> GameRecord:
    return GameRecord(
        pair_id=pair_id,
        game_index=seat,
        candidate_seat=seat,
        map_seed=123,
        outcome=outcome,
        candidate_score=game_score(outcome),
        turns=10,
        elapsed_s=1.0,
        candidate_faults=0,
        opponent_faults=0,
        truncated=False,
        replay_identity=replay_identity(
            candidate_id="cand", opponent_id=opponent, map_seed=123, seat=seat
        ),
    )


def test_store_roundtrip_resume_and_metrics(tmp_path: Path) -> None:
    store = PairedEvalStore(tmp_path / "run")
    assert store.completed_pair_ids() == set()
    store.append_pair(
        PairResult(
            pair_id="opp#0",
            opponent_id="opp",
            map_seed=123,
            candidate_seat_score_a=1.0,
            candidate_seat_score_b=0.0,
            pair_score=0.5,
            games=[_game("opp#0", 0, "WIN"), _game("opp#0", 1, "LOSS")],
        )
    )
    store.append_pair(
        PairResult(
            pair_id="other#0",
            opponent_id="other",
            map_seed=124,
            candidate_seat_score_a=1.0,
            candidate_seat_score_b=1.0,
            pair_score=1.0,
            games=[_game("other#0", 0, "WIN", "other"), _game("other#0", 1, "WIN", "other")],
        )
    )
    reloaded = PairedEvalStore(tmp_path / "run")
    assert reloaded.completed_pair_ids() == {"opp#0", "other#0"}
    pairs = reloaded.load_pairs()
    assert [p.pair_id for p in pairs] == ["opp#0", "other#0"]
    assert reloaded.difference_stream({"opp#0": 0.5, "other#0": 0.25}) == [0.0, 0.75]
    metrics = reloaded.matchup_metrics()
    assert metrics["WORST_MATCHUP_SCORE"] == 0.5
    assert metrics["PAIR_COUNT"] == 2.0
    reloaded.write_summary({"status": "TEST"})
    assert (tmp_path / "run/summary.json").read_text(encoding="utf-8").strip()
    assert '"status": "TEST"' in (tmp_path / "run" / "summary.json").read_text()


def test_store_ignores_partial_trailing_line(tmp_path: Path) -> None:
    store = PairedEvalStore(tmp_path / "run")
    store.append_pair(
        PairResult(
            pair_id="opp#0",
            opponent_id="opp",
            map_seed=123,
            candidate_seat_score_a=0.0,
            candidate_seat_score_b=0.0,
            pair_score=0.0,
            games=[],
        )
    )
    with store.results_path.open("a", encoding="utf-8") as handle:
        handle.write('{"pair_id": "opp#1", "truncated_json')
    assert PairedEvalStore(tmp_path / "run").completed_pair_ids() == {"opp#0"}
