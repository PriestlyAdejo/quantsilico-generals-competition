"""Telemetry must not change actions."""

from __future__ import annotations

import numpy as np

from generals_bot.controls import TelemetryState, control_off, passive_telemetry_noninterference


def test_telemetry_noninterference() -> None:
    logits = np.arange(10, dtype=np.float64)
    mask = np.ones(10, dtype=bool)
    tel = TelemetryState()
    out_l, out_m = passive_telemetry_noninterference(logits, mask, tel)
    assert np.array_equal(out_l, logits)
    assert np.array_equal(out_m, mask)
    assert len(tel.records) == 1
    l2, m2 = control_off(logits, mask)
    assert np.array_equal(l2, logits)
    assert np.array_equal(m2, mask)
