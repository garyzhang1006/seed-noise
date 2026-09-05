"""Populations with a known ``Sigma_E``, which is what makes the identities testable.

The generator here is deliberately not the item-level simulator: it builds half
scores directly from a covariance the caller chooses, so a test can assert that
the estimator returns that exact matrix.  ``seednoise.simulate`` is tested
separately against this one.
"""

from __future__ import annotations

import numpy as np
import pytest

from seednoise.population import MARGIN, Phenotype, Population


def gaussian_population(n_config=400, n_runs=3, K=6, rbar=0.0, sd=None,
                        item_noise=0.4, mu_sd=3.0, seed=0, batch=None,
                        aux_offset_sd=0.0, n_recipes=17, n_sizes=5):
    """Half scores with seed covariance ``Sigma_E`` known in closed form.

    ``item_noise`` is the per-half sampling noise, drawn independently for the two
    halves, which is the assumption the cross-half identity rests on.  An
    ``aux_offset_sd`` above zero adds a run-level offset shared by every auxiliary
    run, the confound the batch-free contrast is built to remove.
    """
    rng = np.random.default_rng(seed)
    sd = np.ones(K) if sd is None else np.asarray(sd, dtype=float)
    C = np.full((K, K), float(rbar))
    np.fill_diagonal(C, 1.0)
    Sig = np.outer(sd, sd) * C
    L = np.linalg.cholesky(Sig + 1e-12 * np.eye(K))

    E = rng.standard_normal((n_config, n_runs, K)) @ L.T
    mu = rng.standard_normal((n_config, 1, K)) * mu_sd
    A = mu + E + rng.standard_normal((n_config, n_runs, K)) * item_noise
    B = mu + E + rng.standard_normal((n_config, n_runs, K)) * item_noise

    if batch is None:
        batch = np.tile(np.arange(n_runs), (n_config, 1))
    else:
        batch = np.tile(np.asarray(batch), (n_config, 1))
    if aux_offset_sd > 0:
        off = rng.standard_normal(n_config) * aux_offset_sd
        shift = np.where(batch != 0, off[:, None], 0.0)
        A = A + shift[:, :, None]
        B = B + shift[:, :, None]

    pop = Population(
        {MARGIN: Phenotype(MARGIN, A, B)},
        gainA=rng.standard_normal((n_config, n_runs)) * 0.1,
        gainB=rng.standard_normal((n_config, n_runs)) * 0.1,
        batch=batch,
        recipe=np.arange(n_config) % n_recipes,
        size=np.arange(n_config) % n_sizes,
    )
    return pop, Sig


@pytest.fixture
def pop_null():
    """Diagonal ``Sigma_E``: the null the headline statistic is tested against."""
    pop, Sig = gaussian_population(rbar=0.0, seed=1)
    return pop, Sig


@pytest.fixture
def pop_corr():
    """Equicorrelated ``Sigma_E`` at 0.3, well inside the alternative."""
    pop, Sig = gaussian_population(rbar=0.3, seed=2)
    return pop, Sig
