"""The mediators: what they measure, and what the regression does to Lambda."""

from __future__ import annotations

import numpy as np
import pytest

from seednoise.estimator import deviations, estimate
from seednoise.mediation import competence, fit_mediation, residualise
from seednoise.population import MARGIN


def test_competence_leaves_its_own_trait_out():
    """Including trait j would put the outcome inside its own predictor."""
    d = np.zeros((2, 3, 4))
    d[:, :, 0] = 100.0
    x = competence(d, np.ones(4))
    assert np.allclose(x[:, :, 0], 0.0)
    assert np.allclose(x[:, :, 1], 100.0 / 3.0)


def test_competence_standardises_before_averaging():
    """A high-variance benchmark must not become the competence axis by itself."""
    d = np.ones((1, 2, 3))
    x_flat = competence(d, np.ones(3))
    x_scaled = competence(d, np.array([1.0, 10.0, 1.0]))
    assert x_flat[0, 0, 0] == pytest.approx(1.0)
    assert x_scaled[0, 0, 0] == pytest.approx((1.0 / 10.0 + 1.0) / 2.0)


def test_a_degenerate_seed_sd_is_refused_rather_than_dividing_by_zero():
    with pytest.raises(ValueError, match="non-positive or non-finite"):
        competence(np.zeros((2, 3, 2)), np.array([1.0, 0.0]))


def test_unknown_mediation_terms_are_named(pop_null):
    pop, _ = pop_null
    with pytest.raises(ValueError, match="unknown mediation terms"):
        fit_mediation(pop, MARGIN, terms=("x", "q"))


def test_no_terms_leaves_the_population_untouched(pop_null):
    pop, _ = pop_null
    fit = fit_mediation(pop, MARGIN, terms=())
    res = residualise(pop, MARGIN, fit)
    name = [n for n in res.phenotypes if n != MARGIN][0]
    assert np.allclose(res.pheno(name).A, deviations(pop.pheno(MARGIN).A))


def test_the_gain_slope_is_recovered_when_it_is_the_only_signal(pop_null):
    """A planted gain effect must come back at its planted size."""
    pop, _ = pop_null
    rng = np.random.default_rng(0)
    g = rng.standard_normal((pop.N, pop.R))
    pop.gainA = g
    pop.gainB = g
    ph = pop.pheno(MARGIN)
    beta = 0.7
    A = ph.A + beta * g[:, :, None]
    B = ph.B + beta * g[:, :, None]
    fit = fit_mediation(pop.with_phenotype(MARGIN, A, B), MARGIN, terms=("s",))
    assert abs(fit.c.mean() - beta) < 0.05     # unbiased across traits
    assert np.abs(fit.c - beta).max() < 0.15   # per-trait sampling noise
    assert np.isnan(fit.b).all()


def test_a_planted_competence_signal_is_removed_by_residualising():
    """Lambda from a pure common factor must collapse once competence is partialled."""
    from tests.conftest import gaussian_population
    pop, _ = gaussian_population(n_config=400, K=6, rbar=0.5, item_noise=0.2, seed=4)
    before = estimate(pop, MARGIN).lambda_hat
    fit = fit_mediation(pop, MARGIN, terms=("x",))
    after = estimate(residualise(pop, MARGIN, fit),
                     f"{MARGIN}|x").lambda_hat
    assert before > 1.5
    assert after < before


def test_residuals_keep_the_design_and_the_shape(pop_corr):
    pop, _ = pop_corr
    res = residualise(pop, MARGIN, fit_mediation(pop, MARGIN))
    name = f"{MARGIN}|x+s"
    assert res.pheno(name).A.shape == pop.pheno(MARGIN).A.shape
    assert np.array_equal(res.recipe, pop.recipe)
    assert np.array_equal(res.batch, pop.batch)
    assert MARGIN in res.phenotypes          # the original stays available


def test_each_half_is_residualised_with_its_own_predictors(pop_corr):
    """Crossing the predictors would import half B's item noise into half A."""
    pop, _ = pop_corr
    fit = fit_mediation(pop, MARGIN, terms=("x",))
    res = residualise(pop, MARGIN, fit)
    ph, rs = pop.pheno(MARGIN), res.pheno(f"{MARGIN}|x")
    sd = np.sqrt(np.clip(np.diag(
        __import__("seednoise.estimator", fromlist=["sigma_e"]).sigma_e(pop, MARGIN)),
        1e-12, None))
    expected = deviations(ph.A) - fit.b * competence(deviations(ph.A), sd)
    assert np.allclose(rs.A, expected)


def test_the_slope_is_pooled_over_every_contrast_not_fitted_per_configuration(pop_corr):
    pop, _ = pop_corr
    fit = fit_mediation(pop, MARGIN, terms=("x", "s"))
    assert fit.n_obs == pop.N * pop.R
    assert fit.b.shape == (pop.K,) and fit.c.shape == (pop.K,)
    rows = fit.as_rows(pop.traits)
    assert len(rows) == pop.K and rows[0]["terms"] == "x+s"
