"""Population evaluation unit tests."""

from __future__ import annotations

import json
from pathlib import Path

from generals_bot.evaluation.population import pfsp_from_empirical, lightweight_psro


def test_empirical_payoff_manifest_exists_and_not_synthetic() -> None:
    path = Path("experiments/manifests/payoff_population_smoke.json")
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["kind"] == "EMPIRICAL"
    assert data["synthetic"] is False
    assert data["games_total"] > 0
    # Diagonal / missing must not be coerced to zero wins for unplayed — unplayed cells null
    for cell in data["cells"]:
        if cell.get("status") == "MISSING":
            assert cell["score_rate"] is None


def test_pfsp_and_psro_from_empirical() -> None:
    path = Path("experiments/manifests/payoff_population_smoke.json")
    payoff = json.loads(path.read_text(encoding="utf-8"))
    pfsp = pfsp_from_empirical(payoff)
    assert abs(pfsp["sum"] - 1.0) < 1e-6
    assert all(p >= pfsp["probability_floor"] - 1e-9 for p in pfsp["probabilities"])
    psro = lightweight_psro(payoff)
    assert psro["finite_nonnegative"]
    assert abs(psro["meta_sum"] - 1.0) < 1e-6
