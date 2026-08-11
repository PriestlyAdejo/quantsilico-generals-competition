"""Episode-consistent recurrent sequence buffers for truncated BPTT."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RecurrentSequenceWindow:
    """One contiguous episode window; burn-in timesteps initialise hidden only."""

    episode_id: str
    cells: np.ndarray  # [T, ...]
    globs: np.ndarray
    actions: np.ndarray
    old_logp: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    terminated: np.ndarray
    burn_in: int
    policy_version: int

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])

    def loss_mask(self) -> np.ndarray:
        """True where PPO policy loss may be applied (excludes burn-in prefix)."""
        mask = np.ones(self.length, dtype=bool)
        mask[: max(0, int(self.burn_in))] = False
        return mask


def build_windows_from_fragment_arrays(
    *,
    episode_id: str,
    cells: np.ndarray,
    globs: np.ndarray,
    actions: np.ndarray,
    old_logp: np.ndarray,
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    seq_len: int,
    burn_in: int,
    policy_version: int,
) -> list[RecurrentSequenceWindow]:
    """Slice a contiguous fragment into episode-consistent windows (no cross-episode shuffle)."""
    t = int(actions.shape[0])
    windows: list[RecurrentSequenceWindow] = []
    start = 0
    while start < t:
        end = min(start + seq_len, t)
        # Do not cross a terminal into the next episode within one window without split
        term_idx = np.where(terminated[start:end])[0]
        if len(term_idx) and int(term_idx[0]) < (end - start - 1):
            end = start + int(term_idx[0]) + 1
        windows.append(
            RecurrentSequenceWindow(
                episode_id=episode_id,
                cells=cells[start:end],
                globs=globs[start:end],
                actions=actions[start:end],
                old_logp=old_logp[start:end],
                rewards=rewards[start:end],
                values=values[start:end],
                terminated=terminated[start:end],
                burn_in=min(burn_in, end - start),
                policy_version=policy_version,
            )
        )
        start = end
    return windows
