"""The bake-off: five models, one held-out quantity, folds over recipe clusters."""

from __future__ import annotations

import numpy as np
import pytest

from seednoise.baselines import (
    BASELINES, bakeoff, observed_sigma_agg, predict_sigma_agg,
)
from seednoise.population import MARGIN
from tests.conftest import gaussian_population


def test_independence_underpredicts_when_the_traits_are_correlated():
    """P0 is the model the whole paper is arguing against, so it must lose here."""
    pop, _ = gaussian_population(n_config=800, K=10, rbar=0.3, seed=0)
    obs = observed_sigma_agg(pop, MARGIN)
    p0 = predict_sigma_agg(pop, pop, MARGIN, "P0")
    p3 = predict_sigma_agg(pop, pop, MARGIN, "P3")
    assert p0 < obs
    assert abs(np.log(p3 / obs)) < abs(np.log(p0 / obs))


def test_independence_is_right_when_the_traits_really_are_independent():
    pop, _ = gaussian_population(n_config=800, K=10, rbar=0.0, seed=1)
    obs = observed_sigma_agg(pop, MARGIN)
    assert abs(np.log(predict_sigma_agg(pop, pop, MARGIN, "P0") / obs)) < 0.1


def test_a_rank_one_truth_is_captured_by_the_rank_one_model():
    pop, _ = gaussian_population(n_config=800, K=10, rbar=0.4, seed=2)
    obs = observed_sigma_agg(pop, MARGIN)
    assert abs(np.log(predict_sigma_agg(pop, pop, MARGIN, "P1") / obs)) < 0.1


def test_every_model_gets_the_same_held_out_per_trait_scales():
    """Only the correlation may differ between models, or the contest is unfair."""
    train, _ = gaussian_population(n_config=400, K=6, rbar=0.3, seed=3)
    test, _ = gaussian_population(n_config=400, K=6, rbar=0.3, seed=4)
    preds = {m: predict_sigma_agg(train, test, MARGIN, m) for m in BASELINES}
    assert len(set(np.round(list(preds.values()), 12))) == len(preds)
    diag = np.sqrt(np.clip(np.diag(
        __import__("seednoise.estimator", fromlist=["sigma_e"]).sigma_e(test, MARGIN)),
        0, None))
    assert preds["P0"] == pytest.approx(np.sqrt((diag ** 2).sum()) / test.K)


def test_an_unknown_model_is_named():
    pop, _ = gaussian_population(n_config=100, K=4, seed=5)
    with pytest.raises(ValueError, match="unknown baseline"):
        predict_sigma_agg(pop, pop, MARGIN, "P9")


def test_folds_are_over_recipes_so_no_recipe_is_in_both_sets():
    pop, _ = gaussian_population(n_config=340, K=6, rbar=0.3, n_recipes=17, seed=6)
    rows = bakeoff(pop, MARGIN, n_folds=5, seed=0)
    assert {r["model"] for r in rows} == set(BASELINES)
    assert all(r["n_folds_scored"] == 5 for r in rows)


def test_too_few_clusters_for_the_requested_folds_is_refused():
    pop, _ = gaussian_population(n_config=40, K=4, n_recipes=3, seed=7)
    with pytest.raises(ValueError, match="cannot be split"):
        bakeoff(pop, MARGIN, n_folds=5)


def test_the_structured_models_beat_independence_out_of_sample():
    pop, _ = gaussian_population(n_config=340, K=10, rbar=0.3, n_recipes=17, seed=8)
    rows = {r["model"]: r["mse_log_sigma_agg"] for r in bakeoff(pop, MARGIN, seed=0)}
    assert rows["P3"] < rows["P0"]
    assert rows["P1"] < rows["P0"]
