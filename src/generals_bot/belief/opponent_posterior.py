"""Hybrid opponent-style posterior from trajectory features + learned embedding."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from generals_bot.models.heads import NUM_ARCHETYPES, OPPONENT_ARCHETYPES, UNKNOWN_FLOOR
from generals_bot.observation import Observation


@dataclass
class TrajectoryFeatures:
    land_growth: float = 0.0
    army_growth: float = 0.0
    scoreboard_residual: float = 0.0
    visible_attacks: float = 0.0
    castle_evidence: float = 0.0
    contact_timing: float = 0.0
    route_pressure: float = 0.0
    retreat_behaviour: float = 0.0
    deathtouch_behaviour: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.land_growth,
                self.army_growth,
                self.scoreboard_residual,
                self.visible_attacks,
                self.castle_evidence,
                self.contact_timing,
                self.route_pressure,
                self.retreat_behaviour,
                self.deathtouch_behaviour,
            ],
            dtype=np.float32,
        )


@dataclass
class OpponentPosteriorState:
    probs: np.ndarray = field(default_factory=lambda: _uniform_with_floor())
    timeline: list[dict[str, float]] = field(default_factory=list)
    prev_opp_land: int | None = None
    prev_opp_army: int | None = None
    turns_seen: int = 0

    def as_dict(self) -> dict[str, float]:
        return {name: float(self.probs[i]) for i, name in enumerate(OPPONENT_ARCHETYPES)}

    def entropy(self) -> float:
        p = np.clip(self.probs, 1e-8, 1.0)
        return float(-(p * np.log(p)).sum())


def _uniform_with_floor() -> np.ndarray:
    probs = np.full(NUM_ARCHETYPES, (1.0 - UNKNOWN_FLOOR) / (NUM_ARCHETYPES - 1), dtype=np.float64)
    probs[0] = UNKNOWN_FLOOR
    probs /= probs.sum()
    return probs.astype(np.float32)


def extract_trajectory_features(
    observation: Observation,
    state: OpponentPosteriorState,
) -> TrajectoryFeatures:
    """Derive explicit features from visible scoreboard / ownership only (no identity)."""
    feats = TrajectoryFeatures()
    if state.prev_opp_land is not None:
        feats.land_growth = (observation.opp_land - state.prev_opp_land) / 50.0
    if state.prev_opp_army is not None:
        feats.army_growth = (observation.opp_army - state.prev_opp_army) / 200.0
    feats.scoreboard_residual = (observation.opp_army - observation.my_army) / 5000.0
    # Visible opponent structures as weak castle evidence.
    castles = 0
    for r in range(observation.height):
        for c in range(observation.width):
            if observation.owner_grid[r][c] == 2 and observation.type_grid[r][c] == 3:
                castles += 1
    feats.castle_evidence = min(castles, 5) / 5.0
    feats.contact_timing = min(state.turns_seen, 200) / 200.0
    return feats


def blend_posterior(
    features: TrajectoryFeatures,
    learned_probs: Tensor | np.ndarray | None = None,
    *,
    feature_weight: float = 0.35,
) -> np.ndarray:
    """Combine heuristic feature logits with optional learned posterior."""
    f = features.as_array()
    logits = np.zeros(NUM_ARCHETYPES, dtype=np.float64)
    # Heuristic mapping (no identity fingerprinting).
    logits[0] = 1.0  # UNKNOWN prior mass
    logits[1] += 2.0 * max(f[0], 0.0)  # RAPID_EXPANDER
    logits[2] += 2.0 * max(f[3], 0.0) + max(f[6], 0.0)  # EARLY_AGGRESSOR
    logits[3] += 2.0 * max(f[1], 0.0)  # COLLECTOR
    logits[4] += 3.0 * f[4]  # CASTLE_INVESTOR
    logits[5] += 1.5 * max(-f[0], 0.0) + max(-f[6], 0.0)  # DEFENSIVE_TURTLE
    logits[6] += 2.0 * max(f[2], 0.0)  # GENERAL_HUNTER
    logits[7] += 3.0 * f[8]  # DEATHTOUCH_SPECIALIST
    logits[8] += 0.5  # MIXED baseline
    feature_probs = _softmax(logits)
    if learned_probs is None:
        mixed = feature_probs
    else:
        if isinstance(learned_probs, Tensor):
            learned = learned_probs.detach().cpu().numpy().reshape(-1)
        else:
            learned = np.asarray(learned_probs, dtype=np.float64).reshape(-1)
        learned = learned / max(learned.sum(), 1e-8)
        mixed = feature_weight * feature_probs + (1.0 - feature_weight) * learned
    return enforce_unknown_floor(mixed)


def enforce_unknown_floor(probs: np.ndarray) -> np.ndarray:
    out = np.asarray(probs, dtype=np.float64).copy()
    out[0] = max(out[0], UNKNOWN_FLOOR)
    out = np.clip(out, 0.0, None)
    out /= out.sum()
    return out.astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def update_opponent_posterior(
    state: OpponentPosteriorState,
    observation: Observation,
    learned_probs: Tensor | np.ndarray | None = None,
) -> OpponentPosteriorState:
    feats = extract_trajectory_features(observation, state)
    probs = blend_posterior(feats, learned_probs)
    state.probs = probs
    state.timeline.append(state.as_dict())
    state.prev_opp_land = observation.opp_land
    state.prev_opp_army = observation.opp_army
    state.turns_seen += 1
    return state


def tensor_posterior(probs: np.ndarray, device: torch.device | None = None) -> Tensor:
    device = device or torch.device("cpu")
    t = torch.as_tensor(enforce_unknown_floor(probs), dtype=torch.float32, device=device)
    return F.normalize(t, p=1, dim=-1)
