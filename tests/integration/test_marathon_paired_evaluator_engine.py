"""Engine-level smoke test for the marathon paired evaluator.

Runs one real seat-swapped pair (legal_random candidate vs pass_bot opponent)
through the pinned competition protocol, then verifies resume semantics.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("generals")

from generals_bot.evaluation.match import run_python_agent_match  # noqa: E402
from generals_bot.marathon_eval.confidence_sequence import AnytimeBoundedCS  # noqa: E402
from generals_bot.marathon_eval.pairing import pair_schedule  # noqa: E402
from generals_bot.marathon_eval.runner import run_evaluation  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_relative_main_paths_resolve(tmp_path: Path, monkeypatch) -> None:
    """Regression: relative agent paths must not resolve against child cwd."""
    monkeypatch.chdir(REPO)
    result = run_python_agent_match(
        Path("baselines/legal_random/main.py"),
        Path("baselines/pass_bot/main.py"),
        seed=7,
        max_turns=80,
    )
    assert not result.crashed0 and not result.crashed1
    assert result.faults0 == 0 and result.faults1 == 0
    # A passing opponent can never capture anything: legal_random wins or
    # the game truncates, it never loses.
    assert result.winner != 1
    assert result.turns > 0


@pytest.mark.integration
def test_one_real_seat_swapped_pair(tmp_path: Path) -> None:
    schedules = pair_schedule(
        eval_namespace="marathon-eval-smoke", opponent_id="pass_bot", pair_count=1
    )
    completed: list[str] = []
    results = run_evaluation(
        run_dir=tmp_path / "smoke",
        schedules=schedules,
        candidate_main=REPO / "baselines/legal_random/main.py",
        opponent_main=REPO / "baselines/pass_bot/main.py",
        candidate_id="legal_random",
        max_turns=80,
        on_pair=lambda pair, seconds: completed.append(pair.pair_id),
    )
    assert completed == ["pass_bot#0"]
    assert len(results) == 1
    pair = results[0]
    assert len(pair.games) == 2
    assert pair.games[0].candidate_seat == 0
    assert pair.games[1].candidate_seat == 1
    assert pair.games[0].map_seed == pair.games[1].map_seed == schedules[0].map_seed
    # legal_random must score strictly above a bot that always passes.
    assert pair.pair_score > 0.0
    assert pair.games[0].candidate_faults == 0
    assert pair.games[1].candidate_faults == 0
    assert pair.games[0].replay_identity != pair.games[1].replay_identity

    # Resume semantics: rerunning the same schedule adds no new pairs.
    resumed = run_evaluation(
        run_dir=tmp_path / "smoke",
        schedules=schedules,
        candidate_main=REPO / "baselines/legal_random/main.py",
        opponent_main=REPO / "baselines/pass_bot/main.py",
        candidate_id="legal_random",
        max_turns=80,
    )
    assert len(resumed) == 1


@pytest.mark.integration
def test_cs_on_real_pair_difference(tmp_path: Path) -> None:
    schedules = pair_schedule(
        eval_namespace="marathon-eval-smoke", opponent_id="pass_bot", pair_count=1
    )
    results = run_evaluation(
        run_dir=tmp_path / "cs",
        schedules=schedules,
        candidate_main=REPO / "baselines/legal_random/main.py",
        opponent_main=REPO / "baselines/pass_bot/main.py",
        candidate_id="legal_random",
        max_turns=80,
    )
    cs = AnytimeBoundedCS(alpha=0.05)
    differences = [pair.pair_score - 0.0 for pair in results]  # pass_bot scores 0
    total = sum(differences)
    interval = cs.interval_on_difference(count=len(differences), difference_total=total)
    assert interval is not None
    lower, upper = interval
    assert lower <= 0.0 <= upper or lower > 0.0  # valid interval either way
    assert upper <= 1.0 and lower >= -1.0
