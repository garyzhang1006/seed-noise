"""The generator has to reproduce the properties the estimator is judged against."""

from __future__ import annotations

import numpy as np
import pytest

from seednoise.estimator import estimate
from seednoise.population import ACCURACY, MARGIN
from seednoise.simulate import SimSpec, default_spec, equicorrelated, simulate

FAST = tuple([400] * 10)


def _spec(**kw):
    kw.setdefault("n_config", 120)
    s = default_spec(**kw)
    s.n_items = FAST
    return s


def test_equicorrelated_is_a_covariance_with_the_requested_correlation():
    S = equicorrelated(10, 0.2, sd=np.full(10, 2.0))
    assert np.allclose(np.diag(S), 4.0)
    assert S[0, 1] == pytest.approx(0.8)
    assert np.linalg.eigvalsh(S).min() > 0


def test_a_correlation_below_the_definiteness_bound_is_refused():
    with pytest.raises(ValueError, match="indefinite"):
        equicorrelated(10, -0.5)


def test_the_same_seed_gives_the_same_population():
    a = simulate(_spec(rbar_e=0.1, seed=5))
    b = simulate(_spec(rbar_e=0.1, seed=5))
    assert np.allclose(a.pheno(MARGIN).A, b.pheno(MARGIN).A)
    c = simulate(_spec(rbar_e=0.1, seed=6))
    assert not np.allclose(a.pheno(MARGIN).A, c.pheno(MARGIN).A)


def test_the_design_matches_the_paper():
    pop = simulate(_spec(rbar_e=0.0, seed=0, n_config=85))
    assert (pop.N, pop.R, pop.K) == (85, 3, 10)
    assert pop.n_clusters == 17
    assert sorted(pop.phenotypes) == [ACCURACY, MARGIN]
    assert pop.size_bands.size == 5


def test_accuracy_is_not_a_linear_image_of_the_margin():
    """Half-level Gaussian generation would make the two phenotypes redundant."""
    pop = simulate(_spec(rbar_e=0.2, seed=1))
    m = pop.pheno(MARGIN).A.ravel()
    a = pop.pheno(ACCURACY).A.ravel()
    r = np.corrcoef(m, a)[0, 1]
    assert 0.5 < r < 0.999


def test_accuracy_lands_in_the_unit_interval_and_above_chance():
    pop = simulate(_spec(rbar_e=0.0, seed=2))
    a = pop.pheno(ACCURACY).A
    assert a.min() >= 0.0 and a.max() <= 1.0
    assert 0.3 < a.mean() < 0.9


@pytest.mark.parametrize("rbar", [0.0, 0.3])
def test_lambda_recovers_the_simulated_truth(rbar):
    pop = simulate(_spec(rbar_e=rbar, seed=3, n_config=600))
    target = np.sqrt(1.0 + 9 * rbar)
    assert abs(estimate(pop, MARGIN).lambda_hat - target) < 0.08


def test_a_run_level_gain_inflates_margin_more_than_accuracy():
    pop = simulate(_spec(rbar_e=0.0, seed=4, n_config=400, gain_sd=0.08))
    lm = estimate(pop, MARGIN).lambda_hat
    la = estimate(pop, ACCURACY).lambda_hat
    assert lm > 1.05 and lm > la


def test_a_batch_offset_is_invisible_to_the_batch_free_contrast():
    """A single direction is noisy, so the claim is averaged over several draws."""
    both = np.array([[estimate(simulate(_spec(rbar_e=0.0, seed=100 + i, n_config=400,
                                              batch_offset_sd=0.5)),
                               MARGIN, which=w).lambda_hat
                      for w in ("all", "batch_free")] for i in range(6)])
    assert both[:, 0].mean() > 1.20
    assert abs(both[:, 1].mean() - 1.0) < 0.05



def test_a_sigma_e_that_is_not_a_covariance_is_refused():
    spec = _spec(rbar_e=0.0, seed=0)
    spec.sigma_e = spec.sigma_e - np.eye(10) * 2.0
    with pytest.raises(ValueError, match="not a covariance"):
        simulate(spec)


def test_a_spec_without_a_sigma_e_says_how_to_build_one():
    spec = SimSpec(n_config=10)
    spec.n_items = FAST
    with pytest.raises(ValueError, match="default_spec"):
        simulate(spec)


def test_the_cross_half_seed_deviation_is_shared_but_the_item_noise_is_not():
    """The halves share the seed effect and nothing else, which is the identity."""
    def cross(spec):
        pop = simulate(spec)
        A, B = pop.pheno(MARGIN).A, pop.pheno(MARGIN).B
        dA = A - A.mean(axis=1, keepdims=True)
        dB = B - B.mean(axis=1, keepdims=True)
        return float(np.mean([np.corrcoef(dA[:, :, j].ravel(),
                                          dB[:, :, j].ravel())[0, 1]
                              for j in range(pop.K)]))

    assert cross(_spec(rbar_e=0.0, seed=7, n_config=300)) > 0.3
    quiet = _spec(rbar_e=0.0, sigma_e_scale=1e-6, seed=7, n_config=300)
    assert abs(cross(quiet)) < 0.1
