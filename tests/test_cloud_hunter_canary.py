from scripts.cloud_hunter_canary import summarize


def test_summarize_counts_wdl_without_claiming_missing_results() -> None:
    rows = [{"wdl": "win"}, {"wdl": "draw"}, {"wdl": "loss"}, {"wdl": None}]

    assert summarize(rows) == {"wins": 1, "draws": 1, "losses": 1}
