r"""The cross-half estimator, its contrast decomposition, and the derived numbers.

The whole paper rests on one identity.  With deviations taken within a
configuration and within an item half,

    E[d^A_rj d^B_rk] = (1 - 1/R) Sigma_E(j,k)

for every (j, k), the diagonal included, because the item-sampling noise on half
A is independent of the noise on half B given the run.  Summing over the R runs
and dividing by R - 1 returns Sigma_E exactly.  The R - 1 divisor is that
identity and not a Bessel correction, and the code says so in one place.

Everything here is written as an average over an orthonormal basis of the
mean-zero contrast space, which makes the batch-free estimator the same function
restricted to one basis direction rather than a second implementation.  With
``w`` a unit vector orthogonal to the all-ones vector,

    E[(sum_r w_r d^A_rj)(sum_r w_r d^B_rk)] = Sigma_E(j,k)

with no divisor at all, since the deviation and the raw run score have the same
projection onto any such direction.  Averaging that over the R - 1 directions of
a full basis reproduces the R - 1 divisor above.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from seednoise.population import Population

__all__ = [
    "contrast_basis", "batch_free_direction", "deviations", "half_projections",
    "TU", "Estimate", "estimate", "sigma_e", "correlation", "k_eff", "rbar_e",
    "p_flip", "phenotypic_correlation", "assert_deviation_form",
]


# -- contrast geometry ---------------------------------------------------------


def contrast_basis(R: int) -> np.ndarray:
    """``(R-1, R)`` orthonormal basis of the mean-zero contrast space.

    Any orthonormal basis gives the same estimator, since the statistic depends
    on the projection onto the whole subspace; the Helmert basis is chosen only
    because its first row is the default-versus-rest contrast, which is the one
    the batch instrument needs to remove.
    """
    if R < 2:
        raise ValueError(f"R={R}: no contrast space below two runs")
    W = np.zeros((R - 1, R))
    for i in range(R - 1):
        # Helmert: run i+1 against the mean of the runs after it.
        W[i, i] = (R - i - 1)
        W[i, i + 1:] = -1.0
        W[i] /= np.linalg.norm(W[i])
    return W


def batch_free_direction(batch: np.ndarray) -> np.ndarray:
    """Unit contrast that is orthogonal to the default-versus-aux direction.

    ``batch`` is one configuration's run labels, 0 for the default run and 1, 2,
    ... for the auxiliary batch.  The returned direction lives entirely inside
    the auxiliary batch, so a run-level offset shared by every aux run cancels
    from it exactly, as does an offset carried only by the default run.
    """
    b = np.asarray(batch).ravel()
    R = b.size
    aux = np.flatnonzero(b != 0)
    if aux.size < 2:
        raise ValueError(
            f"batch labels {b.tolist()} give {aux.size} auxiliary runs; the "
            "batch-free contrast needs at least two runs from the same batch"
        )
    # Any mean-zero vector supported on the aux runs is orthogonal to both the
    # all-ones vector and the default-versus-aux contrast.  With exactly two aux
    # runs this direction is unique up to sign, which is the R = 3 case.
    w = np.zeros(R)
    w[aux[0]] = 1.0
    w[aux[1]] = -1.0
    if aux.size > 2:
        # More than two aux runs: use the full within-aux basis instead, so no
        # information inside the batch is thrown away.
        W = np.zeros((aux.size - 1, R))
        sub = contrast_basis(aux.size)
        W[:, aux] = sub
        return W
    return w[None, :] / np.linalg.norm(w)


# -- deviations and projections ------------------------------------------------


def deviations(y: np.ndarray) -> np.ndarray:
    """Within-configuration, within-half deviations; ``y`` is ``(N, R, K)``."""
    y = np.asarray(y, dtype=np.float64)
    return y - y.mean(axis=1, keepdims=True)


def half_projections(y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """``(N, n_dir, K)`` projections of one half's deviations onto ``W``.

    Projecting the raw scores and projecting the deviations give the same answer
    because every row of ``W`` is orthogonal to the all-ones vector, and the code
    projects the deviations so that a caller who passes residuals from the
    mediation step gets what they expect.
    """
    d = deviations(y)
    return np.einsum("dr,nrk->ndk", np.atleast_2d(W), d)


def assert_deviation_form(pop: Population, name: str) -> None:
    """Check the invariant that makes the raw-score slip impossible here.

    Forming ``T_c`` on raw run scores rather than on within-configuration
    deviations inflates the numerator by the squared configuration mean, which is
    the one-symbol slip of Remark 2.  Writing the statistic as a projection onto
    an orthonormal basis of the mean-zero contrast space removes the failure mode
    rather than detecting it, since every basis row annihilates the all-ones
    vector and the projection of the raw scores is therefore already the
    projection of the deviations.

    What is left to check is that property itself, because an edit to
    ``contrast_basis`` that lost mean-zero rows would reintroduce the bug
    silently.  Both the basis and the realised statistic are checked: the rows
    must sum to zero, and the statistic must not move when a per-configuration
    constant is added to every run of both halves.
    """
    ph = pop.pheno(name)
    W = np.atleast_2d(contrast_basis(pop.R))
    row_sums = np.abs(W.sum(axis=1))
    if row_sums.max() > 1e-10:
        raise AssertionError(
            f"contrast basis row {int(np.argmax(row_sums))} sums to "
            f"{W.sum(axis=1)[int(np.argmax(row_sums))]:.3g} rather than zero, so the "
            "configuration mean enters the statistic; see Remark 2"
        )
    T = _T_from(ph.A, ph.B, W, pop.K)
    scale = float(np.abs(ph.A).max()) + 1.0
    shift = np.random.default_rng(0).standard_normal((pop.N, 1, 1)) * scale
    T_shift = _T_from(ph.A + shift, ph.B + shift, W, pop.K)
    s, ss = float(T.sum()), float(T_shift.sum())
    if not np.isclose(ss, s, rtol=1e-8, atol=1e-10 + 1e-8 * abs(s)):
        raise AssertionError(
            f"phenotype {name!r}: sum_c T_c is {s:.6g} but {ss:.6g} after adding a "
            "per-configuration constant to both halves, so the configuration mean "
            "is leaking into T; see Remark 2"
        )


# -- the statistic -------------------------------------------------------------


def _T_from(A: np.ndarray, B: np.ndarray, W: np.ndarray, K: int) -> np.ndarray:
    pA = half_projections(A, W)          # (N, D, K)
    pB = half_projections(B, W)
    gA = pA.mean(axis=2)                 # (N, D), the aggregate deviation
    gB = pB.mean(axis=2)
    return (gA * gB).mean(axis=1)        # average over the D contrast directions


def _U_from(A: np.ndarray, B: np.ndarray, W: np.ndarray, K: int) -> np.ndarray:
    pA = half_projections(A, W)
    pB = half_projections(B, W)
    per_trait = (pA * pB).mean(axis=1)   # (N, K)
    return per_trait.sum(axis=1) / (K ** 2)


@dataclass(frozen=True)
class TU:
    """Per-configuration numerator and denominator, plus what they imply."""

    T: np.ndarray
    U: np.ndarray
    K: int
    n_directions: int

    @property
    def lambda_hat(self) -> float:
        sT, sU = float(self.T.sum()), float(self.U.sum())
        if sU <= 0:
            return float("nan")
        ratio = sT / sU
        # A negative ratio is a real possibility at small N and is reported as
        # nan rather than clipped to 1, which would hide a failed calibration.
        return float(np.sqrt(ratio)) if ratio >= 0 else float("nan")

    @property
    def sigma_agg(self) -> float:
        m = float(self.T.mean())
        return float(np.sqrt(m)) if m >= 0 else float("nan")

    @property
    def sigma_indep(self) -> float:
        m = float(self.U.mean())
        return float(np.sqrt(m)) if m >= 0 else float("nan")


def _tu(pop: Population, name: str, W) -> TU:
    ph = pop.pheno(name)
    W = np.atleast_2d(W)
    return TU(_T_from(ph.A, ph.B, W, pop.K), _U_from(ph.A, ph.B, W, pop.K),
              pop.K, W.shape[0])


def _basis_for(pop: Population, which: str) -> np.ndarray:
    if which == "all":
        return contrast_basis(pop.R)
    if which == "batch_free":
        rows = [batch_free_direction(pop.batch[c]) for c in range(pop.N)]
        first = rows[0]
        if any(r.shape != first.shape or not np.allclose(r, first) for r in rows):
            raise ValueError(
                "configurations disagree on their batch layout, so one shared "
                "batch-free direction does not exist; split the population by "
                "layout and estimate each part separately"
            )
        return first
    raise ValueError(f"which must be 'all' or 'batch_free', got {which!r}")


@dataclass(frozen=True)
class Estimate:
    phenotype: str
    which: str
    lambda_hat: float
    k_eff: float
    rbar_e: float
    sigma_agg: float
    sigma_indep: float
    N: int
    K: int
    n_directions: int
    T: np.ndarray
    U: np.ndarray

    def as_row(self) -> dict:
        return {
            "phenotype": self.phenotype, "contrast": self.which,
            "Lambda": self.lambda_hat, "K_eff": self.k_eff,
            "rbar_E": self.rbar_e, "sigma_agg": self.sigma_agg,
            "sigma_indep": self.sigma_indep, "N": self.N, "K": self.K,
            "df_per_config": self.n_directions,
        }


def estimate(pop: Population, name: str, which: str = "all",
             check: bool = True) -> Estimate:
    """The headline for one phenotype, on all contrasts or the batch-free one."""
    if check:
        assert_deviation_form(pop, name)
    tu = _tu(pop, name, _basis_for(pop, which))
    lam = tu.lambda_hat
    return Estimate(
        phenotype=name, which=which, lambda_hat=lam, k_eff=k_eff(lam, pop.K),
        rbar_e=rbar_e(lam, pop.K), sigma_agg=tu.sigma_agg,
        sigma_indep=tu.sigma_indep, N=pop.N, K=pop.K,
        n_directions=tu.n_directions, T=tu.T, U=tu.U,
    )


# -- derived quantities --------------------------------------------------------


def k_eff(lam: float, K: int) -> float:
    """``K / Lambda^2``: how many independent benchmarks the battery acts like."""
    if not np.isfinite(lam) or lam <= 0:
        return float("nan")
    return float(K) / float(lam) ** 2


def rbar_e(lam: float, K: int) -> float:
    """``(Lambda^2 - 1)/(K - 1)``, the seed-variance-weighted mean correlation."""
    if not np.isfinite(lam) or K < 2:
        return float("nan")
    return (float(lam) ** 2 - 1.0) / (K - 1)


def p_flip(delta, sd_agg: float) -> np.ndarray:
    """Probability two single-seed runs with true aggregate gap ``delta`` invert."""
    d = np.abs(np.asarray(delta, dtype=np.float64))
    if not np.isfinite(sd_agg) or sd_agg <= 0:
        return np.zeros_like(d)
    return norm.cdf(-d / (np.sqrt(2.0) * sd_agg))


# -- the matrix ----------------------------------------------------------------


def sigma_e(pop: Population, name: str, which: str = "all") -> np.ndarray:
    """``(K, K)`` estimate of ``Sigma_E``, symmetrised over the two half orderings.

    The raw cross-half product is not symmetric in finite samples, and the two
    orderings are equally valid estimates of the same quantity, so the average of
    the two is reported.  Their difference is the statistic gate G3 checks.
    """
    ph = pop.pheno(name)
    W = np.atleast_2d(_basis_for(pop, which))
    pA = half_projections(ph.A, W)
    pB = half_projections(ph.B, W)
    S = np.einsum("ndj,ndk->jk", pA, pB) / (pop.N * W.shape[0])
    return 0.5 * (S + S.T)


def half_asymmetry(pop: Population, name: str, per_config: bool = False):
    """``Cov(d^A_j, d^B_k) - Cov(d^B_j, d^A_k)``, which is zero under G3.

    With ``per_config`` the ``(N, K, K)`` per-configuration contributions come back
    instead of their mean, which is what gate G3 needs to know whether an observed
    asymmetry is larger than the one sampling noise produces at this ``N``.
    """
    ph = pop.pheno(name)
    W = contrast_basis(pop.R)
    pA = half_projections(ph.A, W)
    pB = half_projections(ph.B, W)
    S = np.einsum("ndj,ndk->njk", pA, pB) / W.shape[0]
    D = S - np.swapaxes(S, 1, 2)
    return D if per_config else D.mean(axis=0)


def nearest_psd(S: np.ndarray, floor: float = 0.0):
    """Closest positive semi-definite matrix in the Frobenius sense, and the mass clipped.

    An unbiased cross-half covariance estimate need not be a covariance, since a
    small eigenvalue can land below zero by chance.  Anything that has to sample
    from the estimate, as null N4 does, needs a real covariance, and the size of
    the clipped mass is reported so a badly indefinite estimate is visible rather
    than silently repaired.
    """
    S = np.asarray(S, dtype=np.float64)
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    clipped = float(-w[w < floor].sum() + floor * (w < floor).sum())
    w = np.clip(w, floor, None)
    P = (V * w) @ V.T
    return 0.5 * (P + P.T), clipped


def correlation(S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Correlation matrix of a covariance estimate whose diagonal may be negative.

    A cross-half diagonal entry is an unbiased variance estimate and can come out
    negative by chance, in which case its correlations are undefined and are
    reported as nan rather than as a repaired number.
    """
    S = np.asarray(S, dtype=np.float64)
    d = np.diag(S).copy()
    bad = d <= eps
    sd = np.sqrt(np.where(bad, np.nan, d))
    return S / np.outer(sd, sd)


def phenotypic_correlation(pop: Population, name: str) -> np.ndarray:
    """``R_P`` from the configuration means, the free matrix of the Cheverud question."""
    ph = pop.pheno(name)
    mu = 0.5 * (ph.A.mean(axis=1) + ph.B.mean(axis=1))   # (N, K)
    if mu.shape[0] < 3:
        raise ValueError(
            f"R_P needs at least three configurations, got {mu.shape[0]}"
        )
    return np.corrcoef(mu, rowvar=False)
