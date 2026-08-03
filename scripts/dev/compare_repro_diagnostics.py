"""Compare frozen v2f vs 9qd Expander repro manifests game-by-game."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name: str) -> dict:
    return json.loads((ROOT / "experiments" / "manifests" / name).read_text(encoding="utf-8"))


def games(doc: dict, policy: str) -> list[dict]:
    pol = doc.get("policies", {}).get(policy, {})
    if "games" in pol:
        return list(pol["games"])
    out = []
    for g in doc.get("games", []):
        if g.get("policy") == policy or g.get("candidate") == policy:
            out.append(g)
    return out if out else list(doc.get("games", []))


def summarize(label: str, doc: dict, policy: str) -> None:
    s = doc["policies"][policy]["summary"]
    print(f"=== {label} ===")
    print(
        f"W/D/L {s['wins']}/{s['draws']}/{s['losses']} "
        f"disc={s['enemy_general_discovery_rate']:.3f} "
        f"post={s['post_discovery_win_rate']}"
    )
    print("failures", s.get("failure_classes"))
    gs = games(doc, policy)
    print(f"games n={len(gs)}")
    for i, g in enumerate(gs):
        ex = g.get("extras") or {}
        res = g.get("result") or g.get("outcome") or "?"
        print(
            f"  g{i}: {res} turn={g.get('terminal_turn')} "
            f"contact={g.get('first_contact_turn')}-{g.get('last_contact_turn')} "
            f"disc={g.get('enemy_general_discovered')}@{g.get('discovery_turn')} "
            f"fog={ex.get('unresolved_fog_regions', ex.get('unscouted_regions'))} "
            f"mask={g.get('candidate_general_cells_terminal')} "
            f"phase={ex.get('strategic_phase')} "
            f"emerg={ex.get('emergency_activations', ex.get('false_emergency_count'))}"
        )


def main() -> None:
    v2f = load("repro_expander_heuristic_v2f_best_reference.json")
    q9 = load("repro_expander_heuristic_v2_9qd_latest.json")
    v1 = load("repro_expander_heuristic_v1_reference.json")

    summarize("v2f", v2f, "heuristic_v2f_best_reference")
    print()
    summarize("9qd", q9, "heuristic_v2_9qd_latest")
    print()
    summarize("v1", v1, "heuristic_v1_reference")

    gv = games(v2f, "heuristic_v2f_best_reference")
    gq = games(q9, "heuristic_v2_9qd_latest")
    print()
    print("=== paired v2f vs 9qd ===")
    order = {
        "W": 2,
        "win": 2,
        "WIN": 2,
        "D": 1,
        "draw": 1,
        "DRAW": 1,
        "L": 0,
        "loss": 0,
        "LOSS": 0,
    }
    regress = improve = same = 0
    n = min(len(gv), len(gq))
    for i in range(n):
        a, b = gv[i], gq[i]
        ra = a.get("result") or a.get("outcome")
        rb = b.get("result") or b.get("outcome")
        da = a.get("enemy_general_discovered")
        db = b.get("enemy_general_discovered")
        if ra != rb:
            if order.get(str(rb), -1) < order.get(str(ra), -1):
                mark = "REGRESS"
                regress += 1
            elif order.get(str(rb), -1) > order.get(str(ra), -1):
                mark = "IMPROVE"
                improve += 1
            else:
                mark = "DIFF"
                same += 1
        else:
            mark = "="
            same += 1
        print(
            f"  seed={a.get('seed')} pos={a.get('position')} "
            f"{ra}->{rb} disc {da}->{db} [{mark}]"
        )
    print(f"regress={regress} improve={improve} sameish={same}")


if __name__ == "__main__":
    main()
