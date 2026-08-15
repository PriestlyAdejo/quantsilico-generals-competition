"""Engine-level seat-swapped paired evaluation runner."""

from __future__ import annotations

import time
from pathlib import Path

from generals_bot.evaluation.match import MatchResult, run_python_agent_match
from generals_bot.marathon_eval.pairing import (
    PairSchedule,
    game_score,
    pair_score_from_game_scores,
)
from generals_bot.marathon_eval.store import (
    GameRecord,
    PairedEvalStore,
    PairResult,
    replay_identity,
)


def _outcome_for_candidate(result: MatchResult, candidate_seat: int) -> str:
    if result.winner == -1:
        return "DRAW"
    return "WIN" if result.winner == candidate_seat else "LOSS"


def _game_record(
    *,
    schedule: PairSchedule,
    game_index: int,
    candidate_seat: int,
    candidate_id: str,
    result: MatchResult,
) -> GameRecord:
    outcome = _outcome_for_candidate(result, candidate_seat)
    faults = result.faults0 if candidate_seat == 0 else result.faults1
    candidate_crashed = result.crashed0 if candidate_seat == 0 else result.crashed1
    stderr_tail = result.stderr_tail0 if candidate_seat == 0 else result.stderr_tail1
    attribution = "OK" if faults == 0 and not candidate_crashed else "AGENT_FAULT"
    return GameRecord(
        pair_id=schedule.pair_id,
        game_index=game_index,
        candidate_seat=candidate_seat,
        map_seed=schedule.map_seed,
        outcome=outcome,
        candidate_score=game_score(outcome),
        turns=result.turns,
        elapsed_s=result.elapsed_s,
        candidate_faults=result.faults0 if candidate_seat == 0 else result.faults1,
        opponent_faults=result.faults1 if candidate_seat == 0 else result.faults0,
        truncated=result.truncated,
        replay_identity=replay_identity(
            candidate_id=candidate_id,
            opponent_id=schedule.opponent_id,
            map_seed=schedule.map_seed,
            seat=candidate_seat,
        ),
        attribution=attribution,
        detail=stderr_tail,
    )


def run_pair(
    *,
    schedule: PairSchedule,
    candidate_main: Path,
    opponent_main: Path,
    candidate_id: str,
    mode: str = "competition",
    max_turns: int | None = None,
) -> PairResult:
    """Play one canonical pair: candidate in seat 0, then seat 1, same seed."""
    seat_a = run_python_agent_match(
        candidate_main, opponent_main, seed=schedule.map_seed, mode=mode, max_turns=max_turns
    )
    seat_b = run_python_agent_match(
        opponent_main, candidate_main, seed=schedule.map_seed, mode=mode, max_turns=max_turns
    )
    record_a = _game_record(
        schedule=schedule,
        game_index=0,
        candidate_seat=0,
        candidate_id=candidate_id,
        result=seat_a,
    )
    record_b = _game_record(
        schedule=schedule,
        game_index=1,
        candidate_seat=1,
        candidate_id=candidate_id,
        result=seat_b,
    )
    return PairResult(
        pair_id=schedule.pair_id,
        opponent_id=schedule.opponent_id,
        map_seed=schedule.map_seed,
        candidate_seat_score_a=record_a.candidate_score,
        candidate_seat_score_b=record_b.candidate_score,
        pair_score=pair_score_from_game_scores(
            record_a.candidate_score, record_b.candidate_score
        ),
        games=[record_a, record_b],
    )


def run_evaluation(
    *,
    run_dir: Path,
    schedules: list[PairSchedule],
    candidate_main: Path,
    opponent_main: Path,
    candidate_id: str,
    mode: str = "competition",
    max_turns: int | None = None,
    on_pair: object | None = None,
) -> list[PairResult]:
    """Run all scheduled pairs, resuming past completed pair ids atomically."""
    store = PairedEvalStore(run_dir)
    completed = store.completed_pair_ids()
    results = store.load_pairs()
    for schedule in schedules:
        if schedule.pair_id in completed:
            continue
        started = time.perf_counter()
        pair = run_pair(
            schedule=schedule,
            candidate_main=candidate_main,
            opponent_main=opponent_main,
            candidate_id=candidate_id,
            mode=mode,
            max_turns=max_turns,
        )
        store.append_pair(pair)
        results.append(pair)
        if callable(on_pair):
            on_pair(pair, time.perf_counter() - started)
    return results
