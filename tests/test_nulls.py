"""The six nulls, each with the behaviour the pre-registration claims for it.

Replicate counts here are small enough to run in seconds and large enough that
the assertions are about the location of the null rather than about one draw.
"""

from __future__ import annotations

import numpy as np
import pytest

from seednoise.nulls import (
    n1_calibration, n2_permutation, n3_resplit, n4_zero_mediation,
    n5_gain_artifact, n6_batch_offset, permute_seed_labels, summarise,
)
from seednoise.population import ACCURACY, MARGIN
from seednoise.simulate import default_spec, simulate
from tests.conftest import gaussian_population

FAST = tuple([400] * 10)


def test_summarise_drops_non_finite_values_and_keeps_the_truth():
    s = summarise([1.0, 2.0, np.nan, np.inf], truth=1.0)
    assert s["n"] == 2 and s["mean"] == pytest.approx(1.5)
    assert s["bias"] == pytest.approx(0.5)
    assert summarise([np.nan])["n"] == 0


@pytest.mark.parametrize("rbar", [0.0, 0.2])
def test_n1_returns_the_truth_at_each_point_on_the_curve(rbar):
    rows = n1_calibration(rbars=(rbar,), n_rep=40, n_config=85, seed=0,
                          n_items=FAST)
    for r in rows:
        assert abs(r["mean"] - r["truth"]) < 4 * r["sd"] / np.sqrt(r["n"])
        assert r["sd"] > 0


def test_n1_reports_the_batch_free_contrast_on_its_own_curve():
    rows = n1_calibration(rbars=(0.0,), n_rep=30, n_config=85, seed=1,
                          which=("all", "batch_free"), n_items=FAST)
    got = {(r["phenotype"], r["contrast"]) for r in rows}
    assert got == {(MARGIN, "all"), (MARGIN, "batch_free"),
                   (ACCURACY, "all"), (ACCURACY, "batch_free")}
    free = [r for r in rows if r["contrast"] == "batch_free"][0]
    full = [r for r in rows if r["contrast"] == "all"][0]
    assert free["sd"] > full["sd"]      # one direction instead of R - 1


def test_permutation_keeps_the_cross_half_pairing_of_each_run():
    """Permuting the halves separately would break the identity, not the correlation."""
    pop, _ = gaussian_population(n_config=50, K=4, rbar=0.5, seed=0)
    p = permute_seed_labels(pop, MARGIN, np.random.default_rng(0))
    A, B = pop.pheno(MARGIN).A, pop.pheno(MARGIN).B
    pA, pB = p.pheno(f"{MARGIN}|perm").A, p.pheno(f"{MARGIN}|perm").B
    for c in range(pop.N):
        for j in range(pop.K):
            order = [np.argmin(np.abs(A[c, :, j] - v)) for v in pA[c, :, j]]
            assert np.allclose(pB[c, :, j], B[c, order, j])


def test_n2_destroys_the_correlation_but_keeps_the_marginals():
    pop, _ = gaussian_population(n_config=300, K=10, rbar=0.5, seed=1)
    row = n2_permutation(pop, MARGIN, n_rep=100, seed=0)
    assert row["observed"] > 1.4
    assert abs(row["mean"] - 1.0) < 0.05
    assert row["p_perm"] < 0.05


def test_n3_varies_only_the_item_split():
    """The re-split null needs item-level data, and its spread is split noise alone."""
    def rebuild(split_seed):
        spec = default_spec(rbar_e=0.2, n_config=85, seed=7)
        spec.n_items = FAST
        rng = np.random.default_rng(split_seed)
        halves = [rng.permutation(n) < n // 2 for n in spec.n_items]
        return simulate(spec, halves=halves)

    row = n3_resplit(rebuild, MARGIN, n_rep=8, seed=0)
    assert row["n"] == 8
    assert abs(row["mean"] - np.sqrt(1 + 9 * 0.2)) < 0.15
    assert row["sd"] < 0.15          # the split moves it far less than the data do


def test_n4_gives_the_attenuation_the_regression_causes_on_its_own():
    """Post-mediation Lambda is read against this, never against one."""
    pop, _ = gaussian_population(n_config=85, K=10, rbar=0.3, seed=2)
    row = n4_zero_mediation(pop, MARGIN, n_rep=6, seed=0)
    assert row["lambda_before"]["mean"] > row["lambda_after"]["mean"]
    assert row["psd_clipped_mass"] >= 0.0
    assert row["terms"] == "x+s"


def test_n5_inflates_the_margin_far_more_than_the_accuracy():
    rows = n5_gain_artifact(gain_sd=0.08, n_rep=40, n_config=85, seed=0,
                            n_items=FAST)
    m = [r for r in rows if r["phenotype"] == MARGIN][0]
    a = [r for r in rows if r["phenotype"] == ACCURACY][0]
    assert m["mean"] > 1.05
    assert m["mean"] - 1.0 > 2 * (a["mean"] - 1.0)


def test_n6_manufactures_lambda_that_the_batch_free_contrast_ignores():
    rows = n6_batch_offset(offset_sd=0.5, n_rep=30, n_config=85, seed=0,
                           n_items=FAST)
    full = [r for r in rows if r["phenotype"] == MARGIN
            and r["contrast"] == "all"][0]
    free = [r for r in rows if r["phenotype"] == MARGIN
            and r["contrast"] == "batch_free"][0]
    assert full["mean"] > 1.15
    assert abs(free["mean"] - 1.0) < 0.06
