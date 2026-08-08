from scripts.evaluate_jax_opponent_matrix import behavioural_gate


def _results(*, pass_wins: int = 6, random_decisive: int = 1) -> dict[str, dict]:
    base = {
        "wins": 0,
        "losses": 0,
        "pairs_won_both_seats": 0,
        "illegal_actions": 0,
        "protocol_faults": 0,
    }
    results = {name: dict(base) for name in ("pass", "random", "expander", "hunter")}
    results["pass"]["wins"] = pass_wins
    results["pass"]["pairs_won_both_seats"] = pass_wins // 2
    results["random"]["wins"] = random_decisive
    return results


def test_gate_uses_six_of_eight_total_wins_not_six_pairs() -> None:
    gate = behavioural_gate(_results(pass_wins=6))
    assert gate["status"] == "PASS"
    assert gate["pass_wins"] == 6
    assert gate["pass_pairs_won_both_seats"] == 3


def test_gate_rejects_missing_random_signal_and_faults() -> None:
    no_random = _results(random_decisive=0)
    assert behavioural_gate(no_random)["status"] == "FAIL"
    illegal = _results()
    illegal["hunter"]["illegal_actions"] = 1
    assert behavioural_gate(illegal)["status"] == "FAIL"
