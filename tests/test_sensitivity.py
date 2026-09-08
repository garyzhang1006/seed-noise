"""E5: the checks that vary something arbitrary and watch the headline."""

from __future__ import annotations

import numpy as np
import pytest

from seednoise.build import build_population
from seednoise.estimator import estimate
from seednoise.experiments.e5_sensitivity import (
    estimate_gain_sd, matched_spec, run_gain_calibration, run_leave_one_out,
    run_resplit,
)
from seednoise.population import ACCURACY, MARGIN
from seednoise.simulate import default_spec, simulate
from tests.test_build import _corpus

FAST = tuple([300] * 10)


def _pop(gain_sd=0.0, rbar=0.0, seed=0):
    spec = default_spec(rbar_e=rbar, n_config=60, seed=seed, n_items=FAST)
    spec.gain_sd = gain_sd
    return simulate(spec)


def test_the_gain_estimate_recovers_a_planted_gain_and_is_small_without_one():
    with_gain = estimate_gain_sd(_pop(gain_sd=0.08, seed=1))
    without = estimate_gain_sd(_pop(gain_sd=0.0, seed=1))
    assert 0.05 < with_gain["gain_sd"] < 0.12
    assert without["gain_sd"] < 0.03
    assert without["total_var"] > 0.5 * without["leak_var"]
    assert with_gain["gain_sd"] <= with_gain["gain_sd_raw"]


def test_the_matched_spec_copies_the_data_levels_and_is_a_null():
    pop = _pop(rbar=0.2, seed=2)
    spec = matched_spec(pop, MARGIN, gain_sd=0.0, seed=5)
    assert spec.n_config == pop.N and len(spec.boundary_z) == pop.K
    assert np.allclose(spec.sigma_e, np.diag(np.diag(spec.sigma_e)))
    sim = simulate(spec)
    level_data = 0.5 * (pop.pheno(MARGIN).A.mean(axis=(0, 1))
                        + pop.pheno(MARGIN).B.mean(axis=(0, 1)))
    level_sim = 0.5 * (sim.pheno(MARGIN).A.mean(axis=(0, 1))
                       + sim.pheno(MARGIN).B.mean(axis=(0, 1)))
    # The synthetic corpus puts every trait near the same level, so a
    # correlation would only measure noise; compare the levels directly.
    assert np.max(np.abs(level_sim - level_data)) < 0.1 * np.abs(level_data).mean()
    lam = [estimate(simulate(matched_spec(pop, MARGIN, seed=s)), MARGIN,
                    check=False).lambda_hat for s in range(8)]
    assert abs(np.nanmean(lam) - 1.0) < 0.08


def test_gain_calibration_reports_the_share_at_the_estimated_gain():
    pop = _pop(gain_sd=0.05, seed=3)
    res = run_gain_calibration(pop, grid=(0.02,), n_rep=6, seed=0)
    rows = res["gain"]
    assert {r["spec"] for r in rows} == {"registered", "matched"}
    at = [r for r in rows if r["at_estimated_gain"]]
    assert len(at) == 4                      # two specs, two phenotypes
    assert all(np.isfinite(r["share_of_excess"]) or r["observed_lambda"] <= 1
               for r in rows)


def test_resplit_moves_the_split_and_keeps_the_battery(tmp_path):
    d = _corpus(tmp_path, recipes=("r1", "r2", "r3"))
    traits = ["t0", "t1", "t2"]

    def rebuild(s):
        return build_population(d, traits, split_seed=s)[0]

    rows = run_resplit(rebuild, seeds=(1, 2), n_boot=99)
    assert len(rows) == 4
    assert {r["split_seed"] for r in rows} == {1, 2}
    assert all(f"diag_{t}" in r for r in rows for t in traits)
    a = rebuild(1).pheno(MARGIN).A
    b = rebuild(2).pheno(MARGIN).A
    assert not np.allclose(a, b)


def test_leave_one_out_covers_every_recipe_and_size():
    pop = _pop(rbar=0.2, seed=4)
    rows = run_leave_one_out(pop, n_boot=99)
    kinds = {(r["kind"], r["held"]) for r in rows}
    assert len([k for k in kinds if k[0] == "drop_recipe"]) == pop.n_clusters
    assert len([k for k in kinds if k[0] == "only_size"]) == pop.size_bands.size
    drop = [r for r in rows if r["kind"] == "drop_recipe" and r["phenotype"] == MARGIN]
    assert all(r["N"] == pop.N - (pop.recipe == i).sum() for i, r in enumerate(drop))
    assert all(np.isfinite(r["Lambda"]) for r in rows if r["phenotype"] == ACCURACY)


def test_a_per_trait_boundary_moves_each_trait_level_separately():
    spec = default_spec(rbar_e=0.0, n_config=30, seed=6, n_items=FAST)
    spec.boundary_z = tuple([-2.0] * 5 + [2.0] * 5)
    level = simulate(spec).pheno(MARGIN).A.mean(axis=(0, 1))
    assert (level[:5] > 0).all() and (level[5:] < 0).all()
    with pytest.raises(ValueError):
        spec.boundary_z = (1.0, 2.0)
        simulate(spec)
