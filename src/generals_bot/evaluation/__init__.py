"""Evaluation package."""

from generals_bot.evaluation.match import MatchResult, run_python_agent_match
from generals_bot.evaluation.runner import run_live_like_match

__all__ = ["MatchResult", "run_python_agent_match", "run_live_like_match"]
