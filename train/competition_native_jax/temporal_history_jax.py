"""Legal temporal history knob for STAGE5_CAPACITY_VALUE_R1 T2 (predeclared).

Mode "off" (default) is byte-identical to the canonical rollout path. Mode
"k1" appends the PREVIOUS tick's legal spatial observation as extra input
planes (spatial 8 -> 16 channels); history is zeroed at episode boundaries
and at initialisation. Histories are OF LEGAL OBSERVATIONS ONLY, so the
fog-of-war legality gate (EV-0037) holds structurally.

PPO_SEMANTICS: UNCHANGED. Action selection, legal masks, sampling and serving
are untouched; only the observation tensor width seen by the policy changes,
and only when an experiment explicitly arms the mode.
"""

from __future__ import annotations

_MODE = "off"
_VALID = ("off", "k1")


def set_temporal_history_mode(mode: str) -> None:
    global _MODE
    if mode not in _VALID:
        raise ValueError(f"temporal history mode {mode!r} not in {_VALID}")
    _MODE = mode


def active_temporal_history() -> str:
    return _MODE


HISTORY_PLANES = 8  # K=1 frame of the canonical 8-plane legal spatial obs
