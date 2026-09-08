"""Item-level generation of synthetic populations, which is what the nulls need.

The nulls have to push their populations through byte-identical code, so they are
generated at the item level and reduced by the same function that reduces the
released files.  Generating half-level Gaussian scores directly would be faster
and would silently make the accuracy phenotype a linear image of the margin,
destroying the only thing the accuracy arm is there to measure.

Item margins are drawn as

    m_crji = mu_cji + E_crj + kappa_cr * (mu_cji + E_crj) + eta_crji

with ``mu`` the configuration's per-item difficulty, ``E`` the seed effect with
covariance ``Sigma_E``, ``kappa`` a run-level multiplicative gain used by null N5,
and ``eta`` run-by-item idiosyncratic noise.  Accuracy is the thresholded image,
so the information ratio between the two phenotypes emerges from the shape of the
margin distribution rather than being imposed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seednoise.population import ACCURACY, MARGIN, Phenotype, Population

__all__ = ["SimSpec", "equicorrelated", "simulate", "default_spec"]


@dataclass
class SimSpec:
    """Everything that defines a synthetic population.

    ``n_items`` is per trait and matches the real battery by default, because the
    reliability of each phenotype is a function of item count and a simulation on
    fewer items calibrates a different estimator than the one being run.
    """

    n_config: int = 85
    n_runs: int = 3
    traits: tuple = ("arc_challenge", "arc_easy", "boolq", "csqa", "hellaswag",
                     "mmlu", "openbookqa", "piqa", "socialiqa", "winogrande")
    n_items: tuple = (1172, 2376, 3270, 1221, 10042, 14042, 500, 1838, 1954, 1267)
    sigma_e: np.ndarray | None = None        # (K, K) seed covariance
    item_sd: tuple | None = None             # per-trait sd of eta
    item_df: float = 6.0                     # t degrees of freedom for eta's tails
    item_skew: float = 0.4                   # skew of the per-item difficulty
    boundary_z: float | tuple = -0.385       # standardised location of margin zero, scalar or per trait
    gain_sd: float = 0.0                     # run-level multiplicative gain (N5)
    gain_own_sd: float = 1.0                 # part of the gain covariate that is its own
    batch_offset_sd: float = 0.0             # run-batch additive offset (N6)
    n_recipes: int = 17
    n_sizes: int = 5
    seed: int = 0
    dtype: type = np.float32

    @property
    def K(self) -> int:
        return len(self.traits)


def equicorrelated(K: int, r: float, sd=None) -> np.ndarray:
    """``Sigma_E`` with every off-diagonal correlation equal to ``r``.

    This is the structure the calibration curve sweeps, since ``Lambda`` depends
    on the matrix only through the seed-variance-weighted mean off-diagonal
    correlation and an equicorrelated matrix realises any target exactly.
    """
    if not -1.0 / (K - 1) < r < 1.0:
        raise ValueError(
            f"r={r} is outside ({-1.0 / (K - 1):.4f}, 1) and gives an indefinite "
            f"matrix at K={K}"
        )
    sd = np.ones(K) if sd is None else np.asarray(sd, dtype=np.float64)
    if sd.shape != (K,):
        raise ValueError(f"sd must be ({K},), got {sd.shape}")
    C = np.full((K, K), r)
    np.fill_diagonal(C, 1.0)
    return np.outer(sd, sd) * C


def default_spec(rbar_e: float = 0.0, sigma_e_scale: float = 1.0,
                 **kw) -> SimSpec:
    """Spec with an equicorrelated ``Sigma_E`` at the requested mean correlation."""
    spec = SimSpec(**kw)
    K = spec.K
    spec.sigma_e = equicorrelated(K, rbar_e, sd=np.full(K, sigma_e_scale))
    return spec


def _skewed_normal(rng, shape, skew, dtype):
    """Difficulty draws with a controlled skew, via a shifted lognormal mixture."""
    if abs(skew) < 1e-9:
        return rng.standard_normal(shape).astype(dtype)
    z = rng.standard_normal(shape)
    w = rng.standard_normal(shape)
    a = skew / np.sqrt(1.0 + skew ** 2)
    x = a * np.abs(z) + np.sqrt(1.0 - a ** 2) * w
    x = (x - x.mean()) / x.std()
    return x.astype(dtype)


def simulate(spec: SimSpec, halves=None, return_items: bool = False):
    """Generate one synthetic population and reduce it to half scores.

    ``halves`` is the frozen A/B split as a list of boolean arrays, one per trait;
    a fresh random split is drawn when it is omitted, which is what null N3 varies.
    """
    rng = np.random.default_rng(spec.seed)
    K, N, R = spec.K, spec.n_config, spec.n_runs
    dt = spec.dtype
    if spec.sigma_e is None:
        raise ValueError("spec.sigma_e is None; build the spec with default_spec")
    Sig = np.asarray(spec.sigma_e, dtype=np.float64)
    if Sig.shape != (K, K):
        raise ValueError(f"sigma_e must be ({K},{K}), got {Sig.shape}")
    ev = np.linalg.eigvalsh(Sig)
    if ev.min() < -1e-9:
        raise ValueError(
            f"sigma_e has minimum eigenvalue {ev.min():.3g} and is not a covariance"
        )
    L = np.linalg.cholesky(Sig + np.eye(K) * max(0.0, 1e-12 - ev.min()))

    item_sd = (np.full(K, 10.0) if spec.item_sd is None
               else np.asarray(spec.item_sd, dtype=np.float64))
    if item_sd.shape != (K,):
        raise ValueError(f"item_sd must be ({K},), got {item_sd.shape}")
    # A per-trait boundary lets a matched simulation put each trait's mean margin
    # where the data has it, which is what a multiplicative gain acts on.
    boundary_z = np.broadcast_to(np.asarray(spec.boundary_z, dtype=np.float64), (K,))

    # Seed effects: (N, R, K), i.i.d. across runs within a configuration.
    E = (rng.standard_normal((N, R, K)) @ L.T).astype(dt)

    # Run-level multiplicative gain (null N5) and additive batch offset (null N6).
    gain = (rng.standard_normal((N, R)) * spec.gain_sd).astype(dt)
    batch = np.tile(np.arange(R), (N, 1))
    off = np.zeros((N, R), dtype=dt)
    if spec.batch_offset_sd > 0:
        # One offset per configuration shared by every auxiliary run, which is the
        # structure a later auxiliary batch or a truncated schedule produces.
        shift = (rng.standard_normal(N) * spec.batch_offset_sd).astype(dt)
        off[batch != 0] = np.repeat(shift, (batch != 0).sum(axis=1))

    yA = np.zeros((N, R, K), dtype=np.float64)
    yB = np.zeros((N, R, K), dtype=np.float64)
    aA = np.zeros((N, R, K), dtype=np.float64)
    aB = np.zeros((N, R, K), dtype=np.float64)
    gA = np.zeros((N, R), dtype=np.float64)
    gB = np.zeros((N, R), dtype=np.float64)
    nA = np.zeros(K, dtype=np.int64)
    kept = [] if return_items else None

    for j, n in enumerate(spec.n_items):
        n = int(n)
        if halves is None:
            mask = np.zeros(n, dtype=bool)
            mask[rng.permutation(n)[: n // 2]] = True
        else:
            mask = np.asarray(halves[j], dtype=bool)
            if mask.shape != (n,):
                raise ValueError(
                    f"trait {spec.traits[j]}: half mask is {mask.shape} for {n} items"
                )
        nA[j] = int(mask.sum())
        if nA[j] == 0 or nA[j] == n:
            raise ValueError(
                f"trait {spec.traits[j]}: half A holds {nA[j]} of {n} items, so the "
                "cross-half product has nothing to average on one side"
            )

        # Per-item difficulty, shared by every run of a configuration.  The
        # boundary sits at zero, so the location sets the trait's accuracy.
        mu = (_skewed_normal(rng, (N, 1, n), spec.item_skew, dt) * item_sd[j]
              - boundary_z[j] * item_sd[j])
        # Run-by-item idiosyncratic noise with heavy tails.
        t = rng.standard_t(spec.item_df, size=(N, R, n)).astype(dt)
        t *= np.float32(item_sd[j] * np.sqrt((spec.item_df - 2.0) / spec.item_df))
        m = mu + E[:, :, j : j + 1] + t
        m += (gain[:, :, None] * m) + off[:, :, None]

        corr = (m > 0)
        yA[:, :, j] = m[:, :, mask].mean(axis=2)
        yB[:, :, j] = m[:, :, ~mask].mean(axis=2)
        aA[:, :, j] = corr[:, :, mask].mean(axis=2)
        aB[:, :, j] = corr[:, :, ~mask].mean(axis=2)
        # The gain covariate is a per-byte log-likelihood over all choices, which
        # tracks the run-level scale rather than the gold-minus-distractor gap.
        gA += m[:, :, mask].mean(axis=2) * 0.5
        gB += m[:, :, ~mask].mean(axis=2) * 0.5
        if return_items:
            kept.append(m)

    gA /= K
    gB /= K
    # A sharpness covariate that were an exact function of the battery aggregate
    # would be collinear with the competence proxy and the joint mediation model
    # would not be identified, which is a property of the simulator rather than of
    # the data.  The covariate therefore carries a run-level component of its own,
    # as the real per-byte likelihood does, alongside the shared part above.
    seed_scale = float(np.sqrt(np.mean(np.diag(Sig))))
    own = rng.standard_normal((N, R)) * (spec.gain_own_sd * seed_scale)
    gA = gA + gain + own
    gB = gB + gain + own
    recipe = np.arange(N) % max(1, spec.n_recipes)
    size = np.arange(N) % max(1, spec.n_sizes)
    pop = Population(
        {MARGIN: Phenotype(MARGIN, yA, yB), ACCURACY: Phenotype(ACCURACY, aA, aB)},
        gainA=gA, gainB=gB, batch=batch, recipe=recipe, size=size,
        traits=list(spec.traits), n_items=np.asarray(spec.n_items),
    )
    return (pop, kept) if return_items else pop
