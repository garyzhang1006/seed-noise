r"""Reliability, the information ratio, and the power the design actually has.

Two variance components matter and the cross-half construction separates them
without assuming anything.  The cross-half product gives the seed variance
``Sigma_E(j,j)``; the within-configuration variance of one half gives seed plus
item-sampling noise, since a half's deviation carries both.  Their difference is
the item noise ``v_j``, and the ratio of the two is the reliability of a trait
score for measuring the seed effect.

The information ratio explains why the margin is the better-powered phenotype
without appealing to intuition.  A seed that shifts every item's per-byte margin
by ``delta`` moves the mean margin by ``delta`` and accuracy by ``f(0) delta``,
while item noise is ``s_m^2/n`` for the margin and ``p(1-p)/n`` for accuracy.
Writing ``f(0) = phi(z_0)/s_m`` cancels ``s_m`` and leaves ``p(1-p)/phi(z_0)^2``,
a number the data determines rather than a modelling choice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from seednoise.estimator import deviations, sigma_e
from seednoise.population import Population

__all__ = ["variance_components", "Components", "information_ratio",
           "aggregate_reliability", "power_at", "power_table"]


@dataclass(frozen=True)
class Components:
    """Per-trait seed and item-noise variances, and what they imply."""

    sigma_e2: np.ndarray      # (K,) seed variance from the cross-half product
    noise2: np.ndarray        # (K,) per-half item-sampling variance
    traits: list

    @property
    def reliability(self) -> np.ndarray:
        """``Sigma_E(j,j) / (Sigma_E(j,j) + v_j)`` per trait, on one half."""
        tot = self.sigma_e2 + self.noise2
        return np.where(tot > 0, self.sigma_e2 / np.where(tot > 0, tot, 1.0), np.nan)

    @property
    def vbar(self) -> float:
        """``sum_j v_j / sum_j Sigma_E(j,j)``: the aggregate's noise-to-seed ratio."""
        d = float(self.sigma_e2.sum())
        return float(self.noise2.sum() / d) if d > 0 else float("nan")

    def as_rows(self) -> list:
        return [{"trait": t, "sigma_E2": float(self.sigma_e2[j]),
                 "noise2": float(self.noise2[j]),
                 "reliability": float(self.reliability[j])}
                for j, t in enumerate(self.traits)]


def variance_components(pop: Population, name: str) -> Components:
    """Split each trait's within-configuration variance into seed and item noise.

    ``v_j`` can come out negative when the seed variance is small relative to
    sampling error, and it is reported as measured rather than clipped, because a
    clipped zero would silently claim perfect reliability.
    """
    ph = pop.pheno(name)
    S = sigma_e(pop, name)
    seed = np.diag(S).copy()
    dA, dB = deviations(ph.A), deviations(ph.B)
    # E[d^2] = (1 - 1/R)(Sigma_E + v), averaged over the two halves.
    tot = 0.5 * ((dA ** 2).mean(axis=(0, 1)) + (dB ** 2).mean(axis=(0, 1)))
    tot = tot / (1.0 - 1.0 / pop.R)
    return Components(seed, tot - seed, list(pop.traits))


def information_ratio(p, z0=None) -> np.ndarray:
    """``p(1-p)/phi(z_0)^2``, the margin's information advantage over accuracy.

    ``z_0`` defaults to the standardised boundary implied by the accuracy itself,
    ``Phi^{-1}(1-p)``, which is the Gaussian reading; passing a measured ``z_0``
    from the empirical margin distribution overrides it.
    """
    p = np.asarray(p, dtype=np.float64)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError(
            f"accuracy must lie strictly inside (0, 1); got min {p.min():.4g} and "
            f"max {p.max():.4g}, and a trait at floor or ceiling has no ratio"
        )
    z = norm.ppf(1.0 - p) if z0 is None else np.asarray(z0, dtype=np.float64)
    return p * (1.0 - p) / norm.pdf(z) ** 2


def aggregate_reliability(lam: float, vbar: float) -> float:
    """``rho_g = Lambda^2/(Lambda^2 + vbar)``, quoted at the threshold in use."""
    l2 = float(lam) ** 2
    den = l2 + float(vbar)
    return float(l2 / den) if den > 0 else float("nan")


def power_at(lam_alt: float, null_sd: float, alpha: float = 0.05,
             alt_sd: float | None = None) -> float:
    """Power of the one-sided test of ``Lambda > 1`` against ``lam_alt``.

    ``null_sd`` is the sampling SD of ``Lambda_hat`` under the null, measured by
    null N1 rather than assumed; ``alt_sd`` defaults to it and should be passed
    when the null curve shows the SD growing with the truth.
    """
    crit = 1.0 + norm.ppf(1 - alpha) * float(null_sd)
    sd = float(null_sd if alt_sd is None else alt_sd)
    if sd <= 0:
        return float("nan")
    return float(1.0 - norm.cdf((crit - float(lam_alt)) / sd))


def power_table(null_sd: float, K: int = 10, rbars=(0.05, 0.091, 0.115, 0.20),
                alpha: float = 0.05, alt_sd=None) -> list:
    """Realised power at each pre-registered threshold, in ``Lambda`` units."""
    rows = []
    for r in rbars:
        lam = float(np.sqrt(1.0 + (K - 1) * r))
        sd = None if alt_sd is None else float(alt_sd)
        rows.append({"rbar_E": r, "Lambda": lam,
                     "power": power_at(lam, null_sd, alpha, sd)})
    return rows
