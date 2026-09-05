"""The container refuses the shapes that would corrupt an estimate silently."""

from __future__ import annotations

import numpy as np
import pytest

from seednoise.population import ACCURACY, MARGIN, Phenotype, Population


def _pop(N=6, R=3, K=4, seed=0):
    rng = np.random.default_rng(seed)
    ph = {MARGIN: Phenotype(MARGIN, rng.standard_normal((N, R, K)),
                            rng.standard_normal((N, R, K)))}
    return Population(ph, recipe=np.arange(N) % 3, size=np.arange(N) % 2)


def test_halves_of_different_shapes_are_refused():
    with pytest.raises(ValueError, match="must be scored on the same runs"):
        Phenotype(MARGIN, np.zeros((2, 3, 4)), np.zeros((2, 3, 5)))


def test_a_non_finite_score_is_refused_with_the_reason():
    A = np.zeros((2, 3, 4))
    A[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="partially observed cell"):
        Phenotype(MARGIN, A, np.zeros((2, 3, 4)))


def test_one_run_per_configuration_is_refused():
    ph = {MARGIN: Phenotype(MARGIN, np.zeros((5, 1, 3)), np.zeros((5, 1, 3)))}
    with pytest.raises(ValueError, match="one run has no within"):
        Population(ph)


def test_phenotypes_must_agree_on_shape():
    ph = {MARGIN: Phenotype(MARGIN, np.zeros((5, 3, 3)), np.zeros((5, 3, 3))),
          ACCURACY: Phenotype(ACCURACY, np.zeros((4, 3, 3)), np.zeros((4, 3, 3)))}
    with pytest.raises(ValueError, match="disagree on shape"):
        Population(ph)


def test_asking_for_a_missing_phenotype_lists_the_ones_present():
    with pytest.raises(KeyError, match="margin"):
        _pop().pheno("nonsense")


def test_subset_keeps_every_aligned_array_together():
    pop = _pop(N=6)
    sub = pop.subset([0, 2, 4])
    assert sub.N == 3
    assert np.array_equal(sub.recipe, pop.recipe[[0, 2, 4]])
    assert np.array_equal(sub.size, pop.size[[0, 2, 4]])
    assert sub.config_ids == [0, 2, 4]
    assert np.allclose(sub.pheno(MARGIN).A, pop.pheno(MARGIN).A[[0, 2, 4]])


def test_an_out_of_range_or_empty_subset_is_refused():
    pop = _pop(N=4)
    with pytest.raises(ValueError, match="out of range"):
        pop.subset([0, 9])
    with pytest.raises(ValueError, match="empty"):
        pop.subset([])


def test_a_recipe_drawn_twice_becomes_two_clusters():
    """Sharing a label would make the bootstrap treat the copies as one cluster."""
    pop = _pop(N=6)
    out = pop.subset_clusters([0, 0, 1])
    assert out.n_clusters == 3
    assert out.N == int((pop.recipe == 0).sum()) * 2 + int((pop.recipe == 1).sum())


def test_an_unknown_recipe_in_a_cluster_draw_is_named():
    with pytest.raises(ValueError, match="recipe 99"):
        _pop().subset_clusters([99])


def test_with_phenotype_replaces_only_that_phenotype():
    pop = _pop()
    new = pop.with_phenotype("resid", np.ones((pop.N, pop.R, pop.K)),
                             np.ones((pop.N, pop.R, pop.K)))
    assert sorted(new.phenotypes) == ["margin", "resid"]
    assert np.allclose(new.pheno(MARGIN).A, pop.pheno(MARGIN).A)
    assert np.array_equal(new.recipe, pop.recipe)


def test_summary_counts_the_contrasts_the_estimator_will_use():
    s = _pop(N=6, R=3).summary()
    assert s["contrasts"] == 6 * 2
    assert s["clusters"] == 3 and s["K"] == 4


def test_a_batch_or_gain_array_of_the_wrong_shape_is_refused():
    ph = {MARGIN: Phenotype(MARGIN, np.zeros((5, 3, 2)), np.zeros((5, 3, 2)))}
    with pytest.raises(ValueError, match="batch must be"):
        Population(ph, batch=np.zeros((5, 2)))
    with pytest.raises(ValueError, match="gainA must be"):
        Population(ph, gainA=np.zeros((4, 3)))
