"""The cross-half identity, and every trap the paper names around it.

These tests are the reason to trust every number downstream: if the identity
holds numerically and the deviation guard fires, the rest of the pipeline is
bookkeeping on top of a correct estimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import gaussian_population
from seednoise.estimator import (
    Estimate, assert_deviation_form, batch_free_direction, contrast_basis,
    correlation, deviations, estimate, half_asymmetry, half_projections, k_eff,
    p_flip, phenotypic_correlation, rbar_e, sigma_e,
)
from seednoise.population import MARGIN, Phenotype, Population


# -- contrast geometry ---------------------------------------------------------


@pytest.mark.parametrize("R", [2, 3, 4, 9])
def test_contrast_basis_is_orthonormal_and_mean_zero(R):
    W = contrast_basis(R)
    assert W.shape == (R - 1, R)
    assert np.allclose(W @ W.T, np.eye(R - 1))
    assert np.allclose(W.sum(axis=1), 0.0)


def test_contrast_basis_spans_the_whole_mean_zero_space():
    """The projector onto the basis must equal ``I - J/R``, or contrasts are lost."""
    R = 5
    W = contrast_basis(R)
    P = W.T @ W
    assert np.allclose(P, np.eye(R) - np.ones((R, R)) / R)


def test_contrast_basis_refuses_a_single_run():
    with pytest.raises(ValueError, match="no contrast space"):
        contrast_basis(1)


def test_batch_free_direction_is_the_aux_versus_aux_contrast():
    w = batch_free_direction(np.array([0, 1, 2]))
    assert w.shape == (1, 3)
    assert np.isclose(w[0, 0], 0.0)                 # the default run is excluded
    assert np.allclose(np.abs(w[0, 1:]), 1 / np.sqrt(2))
    assert np.isclose(w.sum(), 0.0)


def test_batch_free_direction_is_orthogonal_to_the_batch_contrast():
    """Orthogonality to the default-versus-aux direction is the whole point."""
    b = np.array([0, 1, 2])
    w = batch_free_direction(b)
    v = np.where(b == 0, 1.0, -0.5)                 # default versus aux
    assert np.allclose(w @ v, 0.0)


def test_batch_free_direction_keeps_every_within_aux_direction():
    W = batch_free_direction(np.array([0, 1, 1, 1]))
    assert W.shape == (2, 4)                        # three aux runs, two df
    assert np.allclose(W[:, 0], 0.0)
    assert np.allclose(W @ W.T, np.eye(2))


def test_batch_free_direction_needs_two_runs_in_one_batch():
    with pytest.raises(ValueError, match="at least two runs from the same batch"):
        batch_free_direction(np.array([0, 1]))


# -- the identity --------------------------------------------------------------


def test_cross_half_product_equals_one_minus_one_over_R_times_sigma_e():
    """``E[d^A_rj d^B_rk] = (1 - 1/R) Sigma_E(j,k)``, the diagonal included."""
    pop, Sig = gaussian_population(n_config=40000, n_runs=3, K=4, rbar=0.3, seed=7)
    ph = pop.pheno(MARGIN)
    dA, dB = deviations(ph.A), deviations(ph.B)
    emp = np.einsum("nrj,nrk->jk", dA, dB) / (pop.N * pop.R)
    assert np.allclose(emp, (1 - 1 / pop.R) * Sig, atol=0.02)


def test_the_R_minus_one_divisor_returns_sigma_e_exactly():
    pop, Sig = gaussian_population(n_config=40000, n_runs=3, K=4, rbar=0.3, seed=8)
    ph = pop.pheno(MARGIN)
    dA, dB = deviations(ph.A), deviations(ph.B)
    est = np.einsum("nrj,nrk->jk", dA, dB) / (pop.N * (pop.R - 1))
    assert np.allclose(0.5 * (est + est.T), Sig, atol=0.02)


def test_a_single_contrast_direction_needs_no_divisor():
    """``E[(w.d^A)(w.d^B)] = Sigma_E`` with no ``R - 1`` in sight."""
    pop, Sig = gaussian_population(n_config=40000, n_runs=3, K=4, rbar=0.3, seed=9)
    ph = pop.pheno(MARGIN)
    w = contrast_basis(3)[:1]
    pA = half_projections(ph.A, w)
    pB = half_projections(ph.B, w)
    emp = np.einsum("ndj,ndk->jk", pA, pB) / pop.N
    assert np.allclose(0.5 * (emp + emp.T), Sig, atol=0.03)


def test_averaging_the_basis_reproduces_the_divisor_form():
    """The two ways of writing the estimator agree to machine precision."""
    pop, _ = gaussian_population(n_config=200, n_runs=4, K=5, rbar=0.2, seed=10)
    ph = pop.pheno(MARGIN)
    W = contrast_basis(pop.R)
    pA, pB = half_projections(ph.A, W), half_projections(ph.B, W)
    by_basis = np.einsum("ndj,ndk->jk", pA, pB) / (pop.N * (pop.R - 1))
    dA, dB = deviations(ph.A), deviations(ph.B)
    by_divisor = np.einsum("nrj,nrk->jk", dA, dB) / (pop.N * (pop.R - 1))
    assert np.allclose(by_basis, by_divisor)


def test_projection_is_the_same_on_raw_scores_and_on_deviations():
    """Every basis row is orthogonal to the all-ones vector, so centring is free."""
    pop, _ = gaussian_population(n_config=50, K=3, seed=11)
    ph = pop.pheno(MARGIN)
    W = contrast_basis(pop.R)
    assert np.allclose(half_projections(ph.A, W),
                       np.einsum("dr,nrk->ndk", W, ph.A))


# -- the headline statistic ----------------------------------------------------


def test_lambda_is_one_when_sigma_e_is_diagonal(pop_null):
    pop, _ = pop_null
    e = estimate(pop, MARGIN)
    assert isinstance(e, Estimate)
    assert e.lambda_hat == pytest.approx(1.0, abs=0.03)
    assert e.rbar_e == pytest.approx(0.0, abs=0.01)
    assert e.k_eff == pytest.approx(pop.K, rel=0.06)


@pytest.mark.parametrize("r", [0.05, 0.2, 0.5])
def test_lambda_recovers_the_equicorrelated_truth(r):
    pop, _ = gaussian_population(n_config=4000, K=10, rbar=r, seed=12)
    e = estimate(pop, MARGIN)
    assert e.lambda_hat == pytest.approx(np.sqrt(1 + 9 * r), rel=0.03)
    assert e.rbar_e == pytest.approx(r, abs=0.02)


def test_lambda_is_invariant_to_the_configuration_means():
    """Only within-configuration contrasts enter, so difficulty cannot leak in."""
    pop, _ = gaussian_population(n_config=300, K=5, rbar=0.2, seed=13)
    base = estimate(pop, MARGIN).lambda_hat
    ph = pop.pheno(MARGIN)
    shift = np.random.default_rng(0).standard_normal((pop.N, 1, pop.K)) * 50
    moved = pop.with_phenotype(MARGIN, ph.A + shift, ph.B + shift)
    assert estimate(moved, MARGIN).lambda_hat == pytest.approx(base, rel=1e-10)


def test_k_eff_and_rbar_are_consistent_with_lambda():
    lam = 1.349                                     # the pre-registered threshold
    assert rbar_e(lam, 10) == pytest.approx(0.0910, abs=5e-4)
    assert k_eff(lam, 10) == pytest.approx(10 / lam ** 2)
    assert k_eff(1.0, 10) == pytest.approx(10.0)


def test_derived_quantities_are_nan_rather_than_wrong_when_lambda_fails():
    assert np.isnan(k_eff(float("nan"), 10))
    assert np.isnan(k_eff(-1.0, 10))
    assert np.isnan(rbar_e(float("nan"), 10))


def test_p_flip_is_a_half_at_zero_gap_and_falls_with_the_gap():
    assert p_flip(0.0, 0.5) == pytest.approx(0.5)
    assert p_flip(1.0, 0.5) < p_flip(0.5, 0.5) < 0.5
    assert np.all(p_flip([0.0, 1.0], 0.0) == 0.0)   # no seed noise, no flips
    assert p_flip(-1.0, 0.5) == pytest.approx(p_flip(1.0, 0.5))


# -- the guard the paper promises ----------------------------------------------


def test_assert_deviation_form_passes_on_a_correct_population(pop_null):
    pop, _ = pop_null
    assert_deviation_form(pop, MARGIN) is None


def test_the_contrast_form_is_immune_to_the_raw_score_slip():
    """Remark 2's disaster cannot happen here: the projection kills the mean."""
    pop, _ = gaussian_population(n_config=400, K=10, rbar=0.0, mu_sd=20.0, seed=26)
    ph = pop.pheno(MARGIN)
    W = contrast_basis(pop.R)
    raw = np.einsum("dr,nrk->ndk", W, ph.A).mean(axis=2)
    cen = np.einsum("dr,nrk->ndk", W, deviations(ph.A)).mean(axis=2)
    assert np.allclose(raw, cen)
    assert estimate(pop, MARGIN).lambda_hat == pytest.approx(1.0, abs=0.08)


def test_guard_fires_when_the_basis_loses_its_mean_zero_rows(monkeypatch):
    """The live failure mode is an edit to contrast_basis, so that is what is tested."""
    import seednoise.estimator as est
    pop, _ = gaussian_population(n_config=200, K=4, mu_sd=20.0, seed=14)
    bad = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])      # no longer mean-zero
    monkeypatch.setattr(est, "contrast_basis", lambda R: bad)
    with pytest.raises(AssertionError, match="rather than zero"):
        est.assert_deviation_form(pop, MARGIN)


def test_guard_fires_when_the_statistic_stops_being_shift_invariant(monkeypatch):
    """A projection that bypasses the basis is caught by the invariance check."""
    import seednoise.estimator as est
    pop, _ = gaussian_population(n_config=200, K=4, mu_sd=20.0, seed=14)
    monkeypatch.setattr(est, "half_projections",
                        lambda y, W: np.asarray(y, dtype=np.float64))
    with pytest.raises(AssertionError, match="leaking into T"):
        est.assert_deviation_form(pop, MARGIN)


def test_estimate_runs_the_guard_by_default(monkeypatch):
    pop, _ = gaussian_population(n_config=50, K=3, seed=15)
    seen = []
    import seednoise.estimator as est
    monkeypatch.setattr(est, "assert_deviation_form",
                        lambda p, n: seen.append(n))
    est.estimate(pop, MARGIN)
    assert seen == [MARGIN]
    est.estimate(pop, MARGIN, check=False)
    assert seen == [MARGIN]


# -- the batch-free contrast ---------------------------------------------------


def test_batch_free_estimator_is_unbiased_for_the_same_lambda():
    pop, _ = gaussian_population(n_config=8000, K=10, rbar=0.2, seed=16)
    full = estimate(pop, MARGIN, which="all").lambda_hat
    free = estimate(pop, MARGIN, which="batch_free").lambda_hat
    assert free == pytest.approx(np.sqrt(1 + 9 * 0.2), rel=0.05)
    assert free == pytest.approx(full, rel=0.08)


def test_batch_free_estimator_is_immune_to_a_shared_auxiliary_offset():
    """A run-level offset common to both aux runs is the confound; it must cancel."""
    clean, _ = gaussian_population(n_config=3000, K=10, rbar=0.1, seed=17)
    dirty, _ = gaussian_population(n_config=3000, K=10, rbar=0.1, seed=17,
                                   aux_offset_sd=1.5)
    assert (estimate(dirty, MARGIN, which="batch_free").lambda_hat
            == pytest.approx(estimate(clean, MARGIN, which="batch_free").lambda_hat,
                             rel=1e-9))
    # The full-contrast estimator has no such immunity, which is why it is not
    # the instrument: the offset is read as genuine cross-benchmark structure.
    assert (estimate(dirty, MARGIN).lambda_hat
            > estimate(clean, MARGIN).lambda_hat + 0.2)


def test_batch_free_reports_one_degree_of_freedom_per_configuration():
    pop, _ = gaussian_population(n_config=100, K=4, seed=18)
    assert estimate(pop, MARGIN, which="batch_free").n_directions == 1
    assert estimate(pop, MARGIN, which="all").n_directions == pop.R - 1


def test_a_mixed_batch_layout_is_refused_rather_than_averaged():
    pop, _ = gaussian_population(n_config=10, K=3, seed=19)
    pop.batch[3] = np.array([1, 2, 0])   # same run count, different layout
    with pytest.raises(ValueError, match="disagree on their batch layout"):
        estimate(pop, MARGIN, which="batch_free")


def test_an_unknown_contrast_set_is_refused():
    pop, _ = gaussian_population(n_config=10, K=3, seed=20)
    with pytest.raises(ValueError, match="must be 'all' or 'batch_free'"):
        estimate(pop, MARGIN, which="every_other_one")


# -- the matrix ----------------------------------------------------------------


def test_sigma_e_recovers_the_true_matrix_including_its_diagonal(pop_corr):
    pop, Sig = gaussian_population(n_config=20000, K=5, rbar=0.3, sd=[1, 2, 0.5, 1, 1.5],
                                   seed=21)
    S = sigma_e(pop, MARGIN)
    assert np.allclose(S, Sig, atol=0.05)
    assert np.allclose(S, S.T)


def test_correlation_of_the_estimate_matches_the_true_correlation():
    pop, Sig = gaussian_population(n_config=20000, K=5, rbar=0.3, sd=[1, 2, 0.5, 1, 1.5],
                                   seed=22)
    R = correlation(sigma_e(pop, MARGIN))
    true = Sig / np.outer(np.sqrt(np.diag(Sig)), np.sqrt(np.diag(Sig)))
    assert np.allclose(R, true, atol=0.03)
    assert np.allclose(np.diag(R), 1.0)


def test_correlation_reports_nan_rather_than_repairing_a_negative_variance():
    S = np.array([[1.0, 0.2], [0.2, -0.01]])
    R = correlation(S)
    assert np.isnan(R[1, 1]) and np.isnan(R[0, 1])
    assert R[0, 0] == pytest.approx(1.0)


def test_half_asymmetry_is_zero_up_to_noise_under_a_symmetric_split():
    pop, _ = gaussian_population(n_config=20000, K=4, rbar=0.3, seed=23)
    D = half_asymmetry(pop, MARGIN)
    assert np.allclose(np.diag(D), 0.0)
    assert np.abs(D).max() < 0.05


def test_phenotypic_correlation_is_the_configuration_level_matrix():
    """``R_P`` picks up between-configuration structure that ``R_E`` must not."""
    rng = np.random.default_rng(24)
    N, R, K = 4000, 3, 4
    common = rng.standard_normal((N, 1, 1))
    mu = common * np.array([1.0, 1.0, 1.0, 1.0]) * 3
    E = rng.standard_normal((N, R, K))              # diagonal Sigma_E by construction
    A = mu + E + rng.standard_normal((N, R, K)) * 0.3
    B = mu + E + rng.standard_normal((N, R, K)) * 0.3
    pop = Population({MARGIN: Phenotype(MARGIN, A, B)},
                     gainA=np.zeros((N, R)), gainB=np.zeros((N, R)),
                     batch=np.tile(np.arange(R), (N, 1)),
                     recipe=np.arange(N) % 17, size=np.arange(N) % 5)
    RP = phenotypic_correlation(pop, MARGIN)
    off = RP[~np.eye(K, dtype=bool)]
    assert off.min() > 0.8                          # configurations move together
    RE = correlation(sigma_e(pop, MARGIN))
    assert np.abs(RE[~np.eye(K, dtype=bool)]).max() < 0.05   # seeds do not


def test_phenotypic_correlation_refuses_too_few_configurations():
    pop, _ = gaussian_population(n_config=2, K=3, seed=25)
    with pytest.raises(ValueError, match="at least three configurations"):
        phenotypic_correlation(pop, MARGIN)
