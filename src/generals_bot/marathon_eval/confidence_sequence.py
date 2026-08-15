"""Anytime-valid confidence sequence for bounded paired differences.

Method (documented per EXECUTION_PLAN section 7.2): a normal-mixture
confidence sequence built on Hoeffding's lemma.  Observations bounded in
[a, b] are sub-Gaussian with variance factor (b-a)^2/4, so for any fixed
mean candidate mu the process

    M_t(mu) = (1 + t sigma^2 rho^2)^(-1/2)
              * exp( rho^2 (S_t - t mu)^2 / (2 (1 + t sigma^2 rho^2)) )

is a nonnegative test supermartingale under H0, where S_t is the running sum,
sigma^2 = (b-a)^2/4, and rho^2 is the mixing prior variance.  Inverting the
anytime-valid test {M_t(mu) >= 1/alpha} yields a confidence sequence that is
simultaneously valid at every (data-dependent) stopping time, including
repeated peeking:

    |mean_t - mu| <= sqrt( (1/t^2) (1/rho^2 + t sigma^2)
                           (2 log(1/alpha) + log(1 + t sigma^2 rho^2)) )

This is a conservative bounded-variable member of the anytime-valid CS
family; the programme default naming is recorded alongside results.  Coverage
at fixed and optional stopping is asserted by the evaluator test suite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def mixture_log_martingale(
    *, count: int, total: float, mu: float, sigma_squared: float, rho_squared: float
) -> float:
    """log M_t(mu) for the closed-form normal-mixture supermartingale.

    Validity input: observations in an interval of width ``2 sqrt(sigma_squared)``
    are sub-Gaussian with variance factor ``sigma_squared`` by Hoeffding's
    lemma, so this process is a nonnegative supermartingale under H0 and its
    inversion is anytime-valid.
    """
    centred = total - count * mu
    denominator = 1.0 + count * sigma_squared * rho_squared
    return (rho_squared * centred * centred) / (2.0 * denominator) - 0.5 * math.log(
        denominator
    )


@dataclass(frozen=True)
class AnytimeBoundedCS:
    """Sequential 1 - alpha confidence sequence for observations in [0, 1]."""

    alpha: float = 0.05
    rho_squared: float = 1.0
    lower_bound: float = 0.0
    upper_bound: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1): {self.alpha}")
        if self.rho_squared <= 0.0:
            raise ValueError(f"rho_squared must be positive: {self.rho_squared}")
        if self.upper_bound <= self.lower_bound:
            raise ValueError("upper_bound must exceed lower_bound")

    @property
    def sigma_squared(self) -> float:
        width = self.upper_bound - self.lower_bound
        return width * width / 4.0

    def update(self, *, count: int, total: float) -> tuple[float, float] | None:
        """Return (lower, upper) on the original scale after ``count`` obs.

        ``total`` is the running sum of the shifted [0, 1] observations.
        Returns None until at least one observation exists.
        """
        if count <= 0:
            return None
        mean = total / count
        spread = 1.0 / self.rho_squared + count * self.sigma_squared
        log_term = 2.0 * math.log(1.0 / self.alpha) + math.log(
            1.0 + count * self.sigma_squared * self.rho_squared
        )
        half_width = math.sqrt(spread * log_term) / count
        lower = max(0.0, mean - half_width)
        upper = min(1.0, mean + half_width)
        return lower, upper

    def interval_on_difference(
        self, *, count: int, difference_total: float
    ) -> tuple[float, float] | None:
        """CS on the paired difference mean, given differences in [-1, 1].

        ``difference_total`` is the running sum of raw paired differences;
        observations are shifted to [0, 1] internally and the resulting
        interval is backtransformed.
        """
        shifted_total = (difference_total + count) / 2.0
        shifted = self.update(count=count, total=shifted_total)
        if shifted is None:
            return None
        return backtransform_interval(shifted)


def backtransform_interval(shifted: tuple[float, float]) -> tuple[float, float]:
    """Map a [0, 1]-scale interval back to the [-1, 1] difference scale."""
    lower, upper = shifted
    return 2.0 * lower - 1.0, 2.0 * upper - 1.0
