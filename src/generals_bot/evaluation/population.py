"""Empirical population evaluation producing real payoff records."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from generals import GeneralsEnv
from generals.core import game

from generals_bot.evaluation.match import make_board, make_transition
from generals_bot.observation import GameContext
from generals_bot.policies.base import TraceLevel
from generals_bot.selector import create_policy
from generals_bot.training.bridge_benchmark import extract_numpy_boards
from generals_bot.training.collect_bc import _action_to_jax, _observation_from_arrays

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_COMMIT = "9e3b9d13cca51caa1bb07db48bb85c9e90ce0462"

HEURISTIC_POPULATION = [
    "pass",
    "legal_random",
    "heuristic_v0",
    "heuristic_v1",
    "heuristic_aggressive",
    "heuristic_defensive",
    "heuristic_castle",
    "heuristic_deathtouch",
]


@dataclass
class EvalPreset:
    name: str
    seeds: int
    paired_positions: bool = True
    games_per_pairing: int = 4
    seed_file: str = "experiments/seeds/train.txt"
    max_turns: int = 200
    wall_clock_s: float | None = 600.0
    label: str = "development"


PRESETS = {
    "population_smoke": EvalPreset(
        name="population_smoke",
        seeds=2,
        games_per_pairing=4,
        max_turns=80,
        wall_clock_s=300.0,
        label="wiring_only",
    ),
    "population_development": EvalPreset(
        name="population_development",
        seeds=6,
        games_per_pairing=12,
        max_turns=150,
        wall_clock_s=1800.0,
        label="development",
    ),
}


def _seed_manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_seeds(path: Path, n: int) -> list[int]:
    return [int(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()][:n]


def _wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - margin) / denom, (centre + margin) / denom)


@dataclass
class PairingRecord:
    policy_a: str
    policy_b: str
    seed_split: str
    seed_manifest_hash: str
    seed: int
    position: int
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    protocol_faults: int = 0
    illegal_actions: int = 0
    latency_ms: list[float] = field(default_factory=list)
    engine_commit: str = ENGINE_COMMIT
    bot_commit: str = ""
    model_ids: dict[str, str] = field(default_factory=dict)
    checkpoint_hashes: dict[str, str] = field(default_factory=dict)
    status: str = "EMPIRICAL"
    winner: int | None = None


def _play_inprocess(
    policy_a: str,
    policy_b: str,
    *,
    seed: int,
    swap: bool,
    max_turns: int,
    bot_commit: str,
) -> PairingRecord:
    env = GeneralsEnv(mode="competition")
    transition = make_transition(env)
    get_obs = game.get_observation
    state = make_board(env, seed)
    h, w = (int(d) for d in state.armies.shape)
    names = [policy_b, policy_a] if swap else [policy_a, policy_b]
    p0 = create_policy(names[0], seed=seed)
    p1 = create_policy(names[1], seed=seed + 1)
    st0 = p0.initial_state(GameContext(0, h, w))
    st1 = p1.initial_state(GameContext(1, h, w))
    latencies: list[float] = []
    faults = 0
    winner = None
    for _ in range(max_turns):
        eng0 = get_obs(state, 0)
        eng1 = get_obs(state, 1)
        t0, o0, a0, _, m0 = extract_numpy_boards(eng0, h, w)
        t1, o1, a1, _, m1 = extract_numpy_boards(eng1, h, w)
        obs0 = _observation_from_arrays(t0, o0, a0, m0)
        obs1 = _observation_from_arrays(t1, o1, a1, m1)
        t_act = time.perf_counter()
        try:
            d0 = p0.act(obs0, st0, deterministic=True, trace=TraceLevel.NONE, deadline=None)
            st0 = d0.new_state
            act0 = d0.action
        except Exception:
            faults += 1
            from generals_bot.action import PASS_ACTION

            act0 = PASS_ACTION
        latencies.append((time.perf_counter() - t_act) * 1000)
        try:
            d1 = p1.act(obs1, st1, deterministic=True, trace=TraceLevel.NONE, deadline=None)
            st1 = d1.new_state
            act1 = d1.action
        except Exception:
            faults += 1
            from generals_bot.action import PASS_ACTION

            act1 = PASS_ACTION
        state, info = transition(state, jnp.stack([_action_to_jax(act0), _action_to_jax(act1)]))
        if bool(info.is_done):
            winner = int(info.winner)
            break

    # Map winner back to policy_a perspective (policy_a is always the evaluated row agent)
    # When swap=False, policy_a is player 0; when swap=True, policy_a is player 1.
    pos = 1 if swap else 0
    wins = draws = losses = 0
    if winner is None:
        draws = 1
    elif winner < 0:
        draws = 1
    else:
        a_won = (winner == 0 and not swap) or (winner == 1 and swap)
        if a_won:
            wins = 1
        else:
            losses = 1

    return PairingRecord(
        policy_a=policy_a,
        policy_b=policy_b,
        seed_split="train",
        seed_manifest_hash="",
        seed=seed,
        position=pos,
        games=1,
        wins=wins,
        draws=draws,
        losses=losses,
        protocol_faults=faults,
        latency_ms=latencies,
        bot_commit=bot_commit,
        winner=winner,
        status="EMPIRICAL",
    )


def run_population_eval(
    *,
    preset_name: str = "population_smoke",
    policies: list[str] | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    preset = PRESETS[preset_name]
    policies = policies or HEURISTIC_POPULATION
    seed_path = REPO_ROOT / preset.seed_file
    seeds = _read_seeds(seed_path, preset.seeds)
    manifest_hash = _seed_manifest_hash(seed_path)
    bot_commit = ""
    try:
        import subprocess

        bot_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        bot_commit = "unknown"

    t0 = time.perf_counter()
    records: list[PairingRecord] = []
    # games_per_pairing with paired positions: each seed contributes 2 games when paired
    games_needed = preset.games_per_pairing
    pairs = [(a, b) for a in policies for b in policies if a != b]
    for a, b in pairs:
        played = 0
        seed_i = 0
        while played < games_needed:
            if preset.wall_clock_s and (time.perf_counter() - t0) > preset.wall_clock_s:
                break
            seed = seeds[seed_i % len(seeds)]
            seed_i += 1
            for swap in (False, True) if preset.paired_positions else (False,):
                if played >= games_needed:
                    break
                rec = _play_inprocess(
                    a, b, seed=seed, swap=swap, max_turns=preset.max_turns, bot_commit=bot_commit
                )
                rec.seed_manifest_hash = manifest_hash
                rec.seed_split = Path(preset.seed_file).stem
                records.append(rec)
                played += 1

    # Aggregate matrix
    labels = list(policies)
    n = len(labels)
    matrix = [[None for _ in range(n)] for _ in range(n)]
    counts = [[0 for _ in range(n)] for _ in range(n)]
    wins_m = [[0 for _ in range(n)] for _ in range(n)]
    draws_m = [[0 for _ in range(n)] for _ in range(n)]
    for rec in records:
        i, j = labels.index(rec.policy_a), labels.index(rec.policy_b)
        counts[i][j] += rec.games
        wins_m[i][j] += rec.wins
        draws_m[i][j] += rec.draws
    cells = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i == j:
                continue
            g = counts[i][j]
            if g == 0:
                cells.append(
                    {
                        "policy_a": a,
                        "policy_b": b,
                        "status": "MISSING",
                        "games": 0,
                        "score_rate": None,
                    }
                )
                continue
            score = (wins_m[i][j] + 0.5 * draws_m[i][j]) / g
            matrix[i][j] = score
            lo, hi = _wilson_interval(wins_m[i][j], g)
            cells.append(
                {
                    "policy_a": a,
                    "policy_b": b,
                    "games": g,
                    "wins": wins_m[i][j],
                    "draws": draws_m[i][j],
                    "losses": g - wins_m[i][j] - draws_m[i][j],
                    "score_rate": score,
                    "wilson_low": lo,
                    "wilson_high": hi,
                    "status": "EMPIRICAL",
                    "seed_split": Path(preset.seed_file).stem,
                    "seed_manifest_hash": manifest_hash,
                    "engine_commit": ENGINE_COMMIT,
                    "bot_commit": bot_commit,
                }
            )

    elapsed = time.perf_counter() - t0
    incomplete = bool(preset.wall_clock_s and elapsed >= preset.wall_clock_s)
    payload = {
        "schema_version": 1,
        "kind": "EMPIRICAL",
        "preset": asdict(preset),
        "labels": labels,
        "matrix": matrix,
        "counts": counts,
        "cells": cells,
        "games_total": sum(counts[i][j] for i in range(n) for j in range(n)),
        "elapsed_s": elapsed,
        "incomplete_wall_clock": incomplete,
        "synthetic": False,
        "note": "Empirical in-process matches; missing cells remain null/MISSING.",
        "records_sample": [asdict(r) for r in records[:20]],
    }
    out_path = out_path or (
        REPO_ROOT / "experiments" / "manifests" / f"payoff_{preset_name}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["path"] = str(out_path)
    return payload


def pfsp_from_empirical(payoff: dict, *, temperature: float = 1.0, floor: float = 0.02) -> dict:
    """Derive PFSP probabilities from empirical development payoff vs a focal agent."""
    labels: list[str] = payoff["labels"]
    # Focus on heuristic_v1 row if present else first
    focal = "heuristic_v1" if "heuristic_v1" in labels else labels[0]
    i = labels.index(focal)
    win_rates = []
    opponents = []
    for j, name in enumerate(labels):
        if j == i:
            continue
        cell = payoff["matrix"][i][j]
        games = payoff["counts"][i][j]
        if cell is None or games == 0:
            # unplayed: neutral prior, not free win
            wr = 0.5
        else:
            # Laplace/Dirichlet-style smoothing
            wr = (payoff["counts"][i][j] and (cell * games) + 1) / (games + 2)
            # cell is score rate; reconstruct wins approx
            wr = (cell * games + 1.0) / (games + 2.0)
        win_rates.append(wr)
        opponents.append(name)
    scores = np.clip(1.0 - np.asarray(win_rates, dtype=np.float64), 1e-6, None)
    scores = scores ** (1.0 / max(temperature, 1e-6))
    probs = scores / scores.sum()
    probs = np.maximum(probs, floor)
    probs = probs / probs.sum()
    out = {
        "schema_version": 1,
        "focal": focal,
        "formula": "softmax-like ((1-wr)^1/T) with Laplace prior and probability floor",
        "temperature": temperature,
        "probability_floor": floor,
        "opponents": opponents,
        "input_win_rates_smoothed": win_rates,
        "probabilities": probs.tolist(),
        "sum": float(probs.sum()),
        "source_payoff": payoff.get("path") or payoff.get("preset"),
        "kind": "EMPIRICAL_PFSP",
    }
    path = REPO_ROOT / "experiments" / "manifests" / "pfsp_empirical.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    out["path"] = str(path)
    return out


def lightweight_psro(payoff: dict) -> dict:
    labels = payoff["labels"]
    n = len(labels)
    # Build restricted game from available empirical cells; fill missing with 0.5
    A = np.full((n, n), 0.5, dtype=np.float64)
    for i in range(n):
        A[i, i] = 0.0
        for j in range(n):
            if i == j:
                continue
            v = payoff["matrix"][i][j]
            if v is not None:
                A[i, j] = float(v)
    # Uniform meta-strategy as documented baseline (not Nash claim)
    meta = np.ones(n) / n
    # Detect simple cycles: i beats j beats k beats i with score>0.55
    cycles = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if len({i, j, k}) < 3:
                    continue
                if A[i, j] > 0.55 and A[j, k] > 0.55 and A[k, i] > 0.55:
                    cycles.append([labels[i], labels[j], labels[k]])
    # Weak matchups for focal heuristic_v1
    focal = labels.index("heuristic_v1") if "heuristic_v1" in labels else 0
    weak = [
        {"opponent": labels[j], "score_rate": float(A[focal, j])}
        for j in range(n)
        if j != focal and A[focal, j] < 0.45
    ]
    out = {
        "schema_version": 1,
        "kind": "LIGHTWEIGHT_PSRO",
        "meta_strategy": {labels[i]: float(meta[i]) for i in range(n)},
        "meta_sum": float(meta.sum()),
        "finite_nonnegative": bool(np.isfinite(meta).all() and (meta >= 0).all()),
        "cycles_sample": cycles[:20],
        "cycle_count": len(cycles),
        "weak_matchups_for_focal": weak,
        "next_br_target": weak[0]["opponent"] if weak else labels[(focal + 1) % n],
        "note": "Uniform meta-strategy documented baseline; not a Nash/exploitability claim.",
    }
    path = REPO_ROOT / "experiments" / "manifests" / "psro_lightweight.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    out["path"] = str(path)
    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="population_smoke", choices=sorted(PRESETS))
    args = p.parse_args()
    payoff = run_population_eval(preset_name=args.preset)
    pfsp = pfsp_from_empirical(payoff)
    psro = lightweight_psro(payoff)
    print(json.dumps({
        "games": payoff["games_total"],
        "path": payoff["path"],
        "incomplete": payoff["incomplete_wall_clock"],
        "pfsp_sum": pfsp["sum"],
        "psro_cycles": psro["cycle_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
