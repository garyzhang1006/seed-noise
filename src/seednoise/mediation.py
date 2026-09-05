r"""Partialling competence and gain out of the seed deviations.

The regression is

    y_crj = a_cj + b_j x^(-j)_cr + c_j s_cr + u_crj

with one pooled slope per trait over all 250 contrasts rather than a
within-configuration fit on two degrees of freedom, and with the competence proxy
``x^(-j)`` and the gain covariate ``s`` taken from the item half opposite the
outcome, so that shared item noise cannot manufacture a slope.

Two halves, two roles, and they must not be mixed up.  The slopes are estimated
across halves, because that is what makes them unbiased.  The residuals are then
formed within a half, using that half's own predictors, because the cross-half
identity needs half A's noise to stay independent of half B's; residualising half
A with half-B predictors would carry half-B noise into half A and bias the
cross-half product of the residuals downward by a term of order ``b`` times the
item-noise variance.  Null N4 measures whatever attenuation the fit still
induces, so a post-partialling ``Lambda`` is read against its own null.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seednoise.estimator import deviations, sigma_e
from seednoise.population import Population

__all__ = ["competence", "MediationFit", "fit_mediation", "residualise"]


def competence(d: np.ndarray, sd_e: np.ndarray) -> np.ndarray:
    """``x^(-j)_cr = (1/(K-1)) sum_{k != j} d_crk / sigmahat_E,k``, shape ``(N, R, K)``.

    Leave-one-trait-out is what keeps the proxy from containing the outcome, and
    the per-trait standardisation is what stops a single high-variance benchmark
    from being the competence axis by itself.
    """
    d = np.asarray(d, dtype=np.float64)
    sd = np.asarray(sd_e, dtype=np.float64)
    K = d.shape[2]
    if sd.shape != (K,):
        raise ValueError(f"sd_e must be ({K},), got {sd.shape}")
    if not np.all(np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError(
            "the per-trait seed SD used to standardise the competence proxy has "
            f"non-positive or non-finite entries: {sd.tolist()}"
        )
    z = d / sd
    tot = z.sum(axis=2, keepdims=True)
    return (tot - z) / (K - 1)


@dataclass(frozen=True)
class MediationFit:
    """Pooled slopes per trait, and which mediators were in the model."""

    b: np.ndarray                # (K,) competence slope, nan when not fitted
    c: np.ndarray                # (K,) gain slope, nan when not fitted
    terms: tuple
    n_obs: int

    def as_rows(self, traits) -> list:
        return [{"trait": t, "b_competence": float(self.b[j]),
                 "c_gain": float(self.c[j]), "terms": "+".join(self.terms)}
                for j, t in enumerate(traits)]


def _design(x_opp, s_opp, terms, j):
    cols = []
    if "x" in terms:
        cols.append(x_opp[:, :, j].ravel())
    if "s" in terms:
        cols.append(s_opp.ravel())
    return np.column_stack(cols) if cols else np.zeros((x_opp[:, :, j].size, 0))


def fit_mediation(pop: Population, name: str, terms=("x", "s")) -> MediationFit:
    """One pooled slope per trait, estimated across halves so it is unbiased.

    Both cross-half orderings are fitted and averaged, since outcome-A-on-B and
    outcome-B-on-A are two unbiased estimates of the same slope and using one
    would throw away half the information.
    """
    terms = tuple(terms)
    bad = set(terms) - {"x", "s"}
    if bad:
        raise ValueError(f"unknown mediation terms {sorted(bad)}; use 'x' and 's'")
    ph = pop.pheno(name)
    K = pop.K
    if not terms:
        return MediationFit(np.zeros(K), np.zeros(K), terms, 0)

    sd = np.sqrt(np.clip(np.diag(sigma_e(pop, name)), 1e-12, None))
    dA, dB = deviations(ph.A), deviations(ph.B)
    sA, sB = deviations(pop.gainA[:, :, None])[:, :, 0], \
        deviations(pop.gainB[:, :, None])[:, :, 0]
    xA, xB = competence(dA, sd), competence(dB, sd)

    b = np.full(K, np.nan)
    c = np.full(K, np.nan)
    n_obs = 0
    for j in range(K):
        coefs = []
        for y, x_opp, s_opp in ((dA, xB, sB), (dB, xA, sA)):
            X = _design(x_opp, s_opp, terms, j)
            yy = y[:, :, j].ravel()
            # The configuration means are already removed by `deviations`, so the
            # intercept is zero by construction and is not refitted here.
            coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
            coefs.append(coef)
            n_obs = yy.size
        m = np.mean(coefs, axis=0)
        k = 0
        if "x" in terms:
            b[j] = m[k]; k += 1
        if "s" in terms:
            c[j] = m[k]
    return MediationFit(b, c, terms, n_obs)


def residualise(pop: Population, name: str, fit: MediationFit,
                out_name: str | None = None) -> Population:
    """Subtract the fitted mediators within each half and return a new population.

    Each half is residualised with its own predictors, which is what keeps the two
    halves' item noise independent and the cross-half identity intact.
    """
    ph = pop.pheno(name)
    sd = np.sqrt(np.clip(np.diag(sigma_e(pop, name)), 1e-12, None))
    dA, dB = deviations(ph.A), deviations(ph.B)
    sA, sB = deviations(pop.gainA[:, :, None])[:, :, 0], \
        deviations(pop.gainB[:, :, None])[:, :, 0]
    xA, xB = competence(dA, sd), competence(dB, sd)

    rA, rB = dA.copy(), dB.copy()
    for j in range(pop.K):
        if "x" in fit.terms:
            rA[:, :, j] -= fit.b[j] * xA[:, :, j]
            rB[:, :, j] -= fit.b[j] * xB[:, :, j]
        if "s" in fit.terms:
            rA[:, :, j] -= fit.c[j] * sA
            rB[:, :, j] -= fit.c[j] * sB
    label = out_name or f"{name}|{'+'.join(fit.terms) or 'none'}"
    return pop.with_phenotype(label, rA, rB)
