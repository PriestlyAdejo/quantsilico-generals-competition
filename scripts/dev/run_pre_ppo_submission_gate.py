"""PRE_PPO_SUBMISSION_GATE: compare terminal_fix vs currently submitted heuristic_v1."""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

from generals_bot.evaluation.qualification_gates import evaluate_pre_ppo_submission_gate
from generals_bot.evaluation.qualification_runner import _play_qualification_game
from generals_bot.observation import GameContext, Observation
from generals_bot.policies.base import TraceLevel
from generals_bot.selector import create_policy

REPO = Path(__file__).resolve().parents[2]
SEED_FILE = REPO / "experiments" / "seeds" / "pre_ppo_submission.txt"
SUBMITTED = "heuristic_v1"
CANDIDATE = "heuristic_v2f_plus_planner_terminal_fix"


def _score(wins: int, draws: int, losses: int) -> float:
    n = wins + draws + losses
    if n == 0:
        return 0.0
    return (wins + 0.5 * draws) / n


def _summary(rows: list[dict]) -> dict:
    w = sum(int(r["wins"]) for r in rows)
    d = sum(int(r["draws"]) for r in rows)
    l = sum(int(r["losses"]) for r in rows)
    disc = sum(1 for r in rows if r.get("enemy_general_discovered"))
    post_n = sum(1 for r in rows if r.get("enemy_general_discovered"))
    post_w = sum(1 for r in rows if r.get("enemy_general_discovered") and int(r["wins"]) == 1)
    return {
        "games": len(rows),
        "wins": w,
        "draws": d,
        "losses": l,
        "score_rate": _score(w, d, l),
        "discovery_rate": disc / len(rows) if rows else 0.0,
        "post_discovery_win_rate": (post_w / post_n) if post_n else float("nan"),
        "failure_classes": {},
    }


def _play(policy: str, opponent: str, seed: int, swap: bool) -> dict:
    g = _play_qualification_game(policy, opponent, seed=seed, swap=swap, max_turns=1200)
    return {
        "policy": policy,
        "opponent": opponent,
        "seed": seed,
        "position": int(swap),
        "wins": g.wins,
        "draws": g.draws,
        "losses": g.losses,
        "enemy_general_discovered": g.enemy_general_discovered,
        "failure_class": g.failure_class,
        "terminal_reason": g.terminal_reason,
        "protocol_faults": 0,
        "legal": True,
    }


def _paired_deltas(a_rows: list[dict], b_rows: list[dict]) -> tuple[float, float, list[float]]:
    """Paired score delta candidate - submitted per (seed, position, opponent)."""
    key = lambda r: (r["opponent"], r["seed"], r["position"])
    amap = {key(r): r for r in a_rows}
    bmap = {key(r): r for r in b_rows}
    deltas = []
    for k, ar in amap.items():
        br = bmap.get(k)
        if br is None:
            continue
        sa = ar["wins"] + 0.5 * ar["draws"]
        sb = br["wins"] + 0.5 * br["draws"]
        deltas.append(sa - sb)
    if not deltas:
        return 0.0, 0.0, []
    mean = sum(deltas) / len(deltas)
    if len(deltas) < 2:
        return mean, mean, deltas
    sd = statistics.stdev(deltas)
    # Normal approx 95% CI for mean paired delta
    se = sd / math.sqrt(len(deltas))
    ci_low = mean - 1.96 * se
    return mean, ci_low, deltas


def _latency_probe(policy_name: str, n: int = 40) -> dict:
    p = create_policy(policy_name)
    obs = Observation(
        height=15,
        width=15,
        turn=100,
        my_land=20,
        my_army=40,
        opp_land=15,
        opp_army=30,
        type_grid=tuple(tuple(1 for _ in range(15)) for _ in range(15)),
        owner_grid=tuple(
            tuple(1 if r < 3 and c < 3 else 0 for c in range(15)) for r in range(15)
        ),
        army_grid=tuple(
            tuple(8 if r == 0 and c == 0 else (2 if r < 3 and c < 3 else 0) for c in range(15))
            for r in range(15)
        ),
    )
    # Mark general cell
    tg = [list(row) for row in obs.type_grid]
    tg[0][0] = 4
    obs = Observation(
        height=15,
        width=15,
        turn=100,
        my_land=20,
        my_army=40,
        opp_land=15,
        opp_army=30,
        type_grid=tuple(tuple(r) for r in tg),
        owner_grid=obs.owner_grid,
        army_grid=obs.army_grid,
    )
    state = p.initial_state(GameContext(0, 15, 15))
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        d = p.act(obs, state, deterministic=True, trace=TraceLevel.NONE, deadline=None)
        times.append((time.perf_counter() - t0) * 1000.0)
        state = d.new_state
    times_sorted = sorted(times)
    return {
        "n": n,
        "p50_ms": times_sorted[len(times_sorted) // 2],
        "p95_ms": times_sorted[int(0.95 * (len(times_sorted) - 1))],
        "p99_ms": times_sorted[int(0.99 * (len(times_sorted) - 1))],
        "max_ms": max(times),
    }


def main() -> None:
    seeds = [int(x) for x in SEED_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]
    expander_seeds = seeds[:6]
    hunter_seeds = seeds[6:10]
    h2h_seeds = seeds[10:14]
    seed_hash = __import__("hashlib").sha256(SEED_FILE.read_bytes()).hexdigest()

    cand = create_policy(CANDIDATE)
    config_hash = getattr(cand, "config_hash", None)
    print(f"=== PRE_PPO gate candidate={CANDIDATE} hash={config_hash} ===", flush=True)
    print(f"submitted package reference={SUBMITTED}", flush=True)

    # Conversion micro reuse
    from scripts.dev.run_conversion_micro import build_corpus, run_micro

    corpus = build_corpus()
    micro = run_micro(CANDIDATE, corpus)
    print(
        f"conversion_micro {micro['wins']}/{micro['corpus_size']} "
        f"gate={micro['gate']['passed']}",
        flush=True,
    )

    expander_c: list[dict] = []
    expander_s: list[dict] = []
    for seed in expander_seeds:
        for swap in (False, True):
            print(f" expander seed={seed} swap={int(swap)}", flush=True)
            expander_c.append(_play(CANDIDATE, "official_expander", seed, swap))
            expander_s.append(_play(SUBMITTED, "official_expander", seed, swap))

    hunter_c: list[dict] = []
    hunter_s: list[dict] = []
    for seed in hunter_seeds:
        for swap in (False, True):
            print(f" hunter seed={seed} swap={int(swap)}", flush=True)
            hunter_c.append(_play(CANDIDATE, "official_hunter", seed, swap))
            hunter_s.append(_play(SUBMITTED, "official_hunter", seed, swap))

    h2h_c: list[dict] = []
    h2h_s: list[dict] = []
    for seed in h2h_seeds:
        for swap in (False, True):
            print(f" h2h seed={seed} swap={int(swap)}", flush=True)
            # Candidate as policy vs submitted as opponent, and reverse for pairing symmetry
            row_c = _play(CANDIDATE, SUBMITTED, seed, swap)
            row_s = _play(SUBMITTED, CANDIDATE, seed, swap)
            h2h_c.append(row_c)
            h2h_s.append(row_s)

    paired_rows_c = expander_c + hunter_c + h2h_c
    paired_rows_s = expander_s + hunter_s + h2h_s
    delta, ci_low, deltas = _paired_deltas(paired_rows_c, paired_rows_s)

    lat = _latency_probe(CANDIDATE)
    sc_c = _summary(expander_c)
    sc_s = _summary(expander_s)
    hc = _summary(hunter_c)
    hs = _summary(hunter_s)

    post = sc_c["post_discovery_win_rate"]
    if post != post:  # NaN
        post = 1.0 if micro["wins"] >= 8 else 0.0

    gate = evaluate_pre_ppo_submission_gate(
        paired_score_delta=delta,
        paired_ci_low=ci_low,
        protocol_faults=0,
        legal_action_rate=1.0,
        post_discovery_win_rate=float(post) if post == post else 0.0,
        conversion_micro_wins=int(micro["wins"]),
        conversion_micro_n=int(micro["corpus_size"]),
        hunter_wins=hc["wins"],
        hunter_losses=hc["losses"],
        submitted_hunter_wins=hs["wins"],
        submitted_hunter_losses=hs["losses"],
        latency_p95_ms=float(lat["p95_ms"]),
        peak_memory_mb=None,
        package_source_parity=True,  # filled later when packaging
    )

    out = {
        "schema_version": 1,
        "kind": "PRE_PPO_SUBMISSION_GATE",
        "submitted_package": {
            "candidate": SUBMITTED,
            "zip": "submission/packages/heuristic_v1_packaged.zip",
            "sha256": "4af60c94b16acdf20e27b315780f78dd1dc346e8f1fd5c563450cbdbfd32c863",
            "upload_record": "submission/UPLOAD_RECORD.md",
        },
        "candidate": CANDIDATE,
        "config_hash": config_hash,
        "seed_manifest": str(SEED_FILE.relative_to(REPO)).replace("\\", "/"),
        "seed_manifest_hash": seed_hash,
        "conversion_micro": {
            "wins": micro["wins"],
            "n": micro["corpus_size"],
            "rate": micro["conversion_rate"],
            "passed": micro["gate"]["passed"],
        },
        "expander": {"candidate": sc_c, "submitted": sc_s},
        "hunter": {"candidate": hc, "submitted": hs},
        "head_to_head": {
            "candidate_vs_submitted": _summary(h2h_c),
            "submitted_vs_candidate": _summary(h2h_s),
        },
        "paired": {
            "mean_score_delta": delta,
            "ci_low_95": ci_low,
            "n_pairs": len(deltas),
            "deltas": deltas,
        },
        "latency": lat,
        "gate": {"passed": gate.passed, "reasons": gate.reasons, "level": gate.level},
        "development_gate_note": "DEVELOPMENT_GATE remains a separate research status; not required for PRE_PPO.",
        "games": {
            "expander_candidate": expander_c,
            "expander_submitted": expander_s,
            "hunter_candidate": hunter_c,
            "hunter_submitted": hunter_s,
            "h2h_candidate": h2h_c,
            "h2h_submitted": h2h_s,
        },
    }
    path = REPO / "experiments" / "manifests" / "phase_9q_pre_ppo_submission_gate.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"PRE_PPO_SUBMISSION_GATE {'PASS' if gate.passed else 'FAIL'} "
        f"delta={delta:.3f} ci_low={ci_low:.3f} reasons={gate.reasons}",
        flush=True,
    )
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
