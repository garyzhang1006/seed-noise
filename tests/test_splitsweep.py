"""E6: the estimation-subset sweep agrees with subsetting the population."""

from __future__ import annotations

import csv
import json
from itertools import combinations
from math import comb

import numpy as np
import pytest

from seednoise.cli import main
from seednoise.estimator import estimate
from seednoise.experiments.e6_splitsweep import (
    recipe_tu, run_splitsweep, seed_partition, sweep_lambda,
)
from seednoise.population import ACCURACY, MARGIN
from seednoise.simulate import default_spec, simulate


@pytest.fixture(scope="module")
def pop():
    spec = default_spec(rbar_e=0.15, n_config=30, seed=11)
    spec.n_recipes = 6
    spec.n_items = tuple(200 for _ in spec.n_items)
    return simulate(spec)


def test_recipe_sums_reproduce_the_full_estimate(pop):
    for name in (MARGIN, ACCURACY):
        codes, T, U, n = recipe_tu(pop, name)
        assert codes.tolist() == list(range(6)) and n.sum() == pop.N
        assert np.sqrt(T.sum() / U.sum()) == pytest.approx(
            estimate(pop, name).lambda_hat)


def test_the_sweep_matches_subsetting_the_population_for_every_subset(pop):
    codes, T, U, _ = recipe_tu(pop, MARGIN)
    lam, lo, hi = sweep_lambda(T, U, 4, chunk=5)      # chunk < 15 subsets
    assert lam.shape == (comb(6, 4),)
    for j, sub in enumerate(combinations(range(6), 4)):
        keep = np.isin(pop.recipe, codes[list(sub)])
        want = estimate(pop.subset(np.flatnonzero(keep)), MARGIN).lambda_hat
        assert lam[j] == pytest.approx(want)
    assert lam[np.argmin(lam)] == pytest.approx(np.sqrt(T[lo].sum() / U[lo].sum()))
    assert lam[np.argmax(lam)] == pytest.approx(np.sqrt(T[hi].sum() / U[hi].sum()))


def test_a_negative_ratio_is_nan_not_a_winner():
    T = np.array([1.0, -5.0, 2.0])
    U = np.array([1.0, 1.0, 1.0])
    lam, lo, hi = sweep_lambda(T, U, 2)
    assert np.isnan(lam[0]) and np.isnan(lam[2])   # {0,1} and {1,2} sum T < 0
    assert lam[1] == pytest.approx(np.sqrt(1.5))
    assert sorted(hi.tolist()) == [0, 2] and sorted(lo.tolist()) == [0, 2]


def test_the_seed_partition_is_deterministic_and_covers_every_recipe():
    names = [f"r{i}" for i in range(25)]
    scr, est = seed_partition(names, 17)
    assert len(scr) == 8 and len(est) == 17 and sorted(scr + est) == sorted(names)
    assert (scr, est) == seed_partition(list(reversed(names)), 17)
    assert est != seed_partition(names, 17, seed=1)[1]


def test_the_rows_carry_the_registered_thresholds_and_a_share(pop):
    e6 = run_splitsweep(pop, n_estimation=4)
    rows = {r["phenotype"]: r for r in e6["splitsweep"]}
    assert rows[MARGIN]["threshold"] == 1.349 and rows[ACCURACY]["threshold"] == 1.40
    for r in rows.values():
        assert r["n_subsets"] == comb(6, 4)
        assert r["min"] <= r["q05"] <= r["q50"] <= r["q95"] <= r["max"]
        assert 0.0 <= r["share_above_threshold"] <= 1.0
        assert r["min"] <= r["Lambda_seed_partition"] <= r["max"]
    assert len(e6["recipes"]) == 6
    assert sum(r["in_seed_estimation_set"] for r in e6["recipes"]) == 4
    assert "never implemented" in e6["source"]["note"]
    with pytest.raises(ValueError, match="recipe names"):
        run_splitsweep(pop, recipe_names=["a", "b"], n_estimation=4)


def test_the_command_writes_the_three_files(tmp_path):
    rc = main(["splitsweep", "--synthetic", "--fast", "--n-config", "34",
               "--n-estimation", "12", "--threshold-margin", "1.2",
               "--out", str(tmp_path), "--quiet"])
    assert rc == 0
    with open(tmp_path / "tab_splitsweep.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["phenotype"] for r in rows] == [MARGIN, ACCURACY]
    assert float(rows[0]["threshold"]) == 1.2 and float(rows[1]["threshold"]) == 1.40
    assert int(rows[0]["n_subsets"]) == comb(17, 12)
    with open(tmp_path / "tab_splitsweep_recipes.csv", newline="") as fh:
        assert len(list(csv.DictReader(fh))) == 17
    src = json.loads((tmp_path / "splitsweep_source.json").read_text())
    assert src["n_subsets"] == comb(17, 12)
    assert len(src["seed_partition"]["estimation"]) == 12
