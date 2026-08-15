"""Atomic, resumable pair-result storage with replay identity."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class GameRecord:
    pair_id: str
    game_index: int
    candidate_seat: int  # 0 or 1
    map_seed: int
    outcome: str  # WIN | DRAW | LOSS from the candidate's perspective
    candidate_score: float
    turns: int
    elapsed_s: float
    candidate_faults: int
    opponent_faults: int
    truncated: bool
    replay_identity: str  # sha256 over inputs sufficient to replay the game
    attribution: str = "OK"  # OK | AGENT_FAULT | EVALUATOR_FAULT
    detail: str = ""  # captured evidence (e.g. agent stderr tail on crash)


@dataclass
class PairResult:
    pair_id: str
    opponent_id: str
    map_seed: int
    candidate_seat_score_a: float
    candidate_seat_score_b: float
    pair_score: float
    games: list[GameRecord] = field(default_factory=list)


class PairedEvalStore:
    """Append-only JSONL results plus atomic pair-level summaries.

    Each completed pair is written as one line followed by an fsync so an
    interrupted evaluation resumes from the last complete pair; partial pairs
    are ignored on load.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.run_dir / "pair_results.jsonl"
        self.summary_path = self.run_dir / "summary.json"

    def append_pair(self, pair: PairResult) -> None:
        record = asdict(pair)
        line = json.dumps(record, sort_keys=True)
        with self.results_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def load_pairs(self) -> list[PairResult]:
        if not self.results_path.exists():
            return []
        pairs = []
        for line in self.results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A truncated trailing line means the pair never completed;
                # atomic append guarantees no completed pair is corrupt.
                continue
            pairs.append(
                PairResult(
                    pair_id=record["pair_id"],
                    opponent_id=record["opponent_id"],
                    map_seed=record["map_seed"],
                    candidate_seat_score_a=record["candidate_seat_score_a"],
                    candidate_seat_score_b=record["candidate_seat_score_b"],
                    pair_score=record["pair_score"],
                    games=[GameRecord(**game) for game in record.get("games", [])],
                )
            )
        return pairs

    def completed_pair_ids(self) -> set[str]:
        return {pair.pair_id for pair in self.load_pairs()}

    def difference_stream(self, incumbent_pair_scores: dict[str, float]) -> list[float]:
        """Paired differences (candidate - incumbent) in stored order."""
        return [
            pair.pair_score - incumbent_pair_scores[pair.pair_id]
            for pair in self.load_pairs()
            if pair.pair_id in incumbent_pair_scores
        ]

    def matchup_metrics(self) -> dict[str, float]:
        """WORST_MATCHUP_SCORE / BOTTOM_QUARTILE_MATCHUP_SCORE / STD_MATCHUP_SCORE."""
        by_opponent: dict[str, list[float]] = {}
        for pair in self.load_pairs():
            by_opponent.setdefault(pair.opponent_id, []).append(pair.pair_score)
        matchup_scores = {
            opponent: sum(scores) / len(scores) for opponent, scores in by_opponent.items()
        }
        if not matchup_scores:
            return {
                "WORST_MATCHUP_SCORE": float("nan"),
                "BOTTOM_QUARTILE_MATCHUP_SCORE": float("nan"),
                "STD_MATCHUP_SCORE": float("nan"),
                "PAIR_COUNT": 0.0,
            }
        values = sorted(matchup_scores.values())
        bottom_count = max(1, len(values) // 4)
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return {
            "WORST_MATCHUP_SCORE": values[0],
            "BOTTOM_QUARTILE_MATCHUP_SCORE": sum(values[:bottom_count]) / bottom_count,
            "STD_MATCHUP_SCORE": variance**0.5,
            "PAIR_COUNT": float(sum(len(scores) for scores in by_opponent.values())),
        }

    def write_summary(self, summary: dict) -> None:
        tmp = self.summary_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.summary_path)


def replay_identity(*, candidate_id: str, opponent_id: str, map_seed: int, seat: int) -> str:
    import hashlib

    material = f"{candidate_id}|{opponent_id}|{map_seed}|{seat}".encode()
    return hashlib.sha256(material).hexdigest()
