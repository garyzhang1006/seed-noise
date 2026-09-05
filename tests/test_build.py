"""Where the item-level and array-level halves of the pipeline meet.

Every property here is one that, if it broke, would produce a population that
looks fine and is wrong: misaligned items, a padded cell, or a run index that
means a different batch in different configurations.
"""

from __future__ import annotations

import numpy as np
import pytest

from seednoise.build import build_population
from seednoise.halves import split_items
from seednoise.phenotypes import ItemPhenotypes
from seednoise.population import ACCURACY, MARGIN
from seednoise.store import save_run

TRAITS = ["a", "b", "c"]
N_ITEMS = (40, 60, 50)


def _items(seed, n_items=N_ITEMS, offset=0):
    rng = np.random.default_rng(seed)
    trait = np.repeat(np.arange(len(n_items)), n_items)
    iid = np.arange(trait.size) + offset
    m = rng.standard_normal(trait.size) + 0.3
    return ItemPhenotypes(iid, trait, m, m > 0, group=trait.copy())


def _corpus(tmp_path, recipes=("r1", "r2"), sizes=("60M", "90M"), seeds=(2, 4, 5),
            drop=None, n_items=N_ITEMS, offset_for=None):
    d = tmp_path / "runs"
    for rc in recipes:
        for sz in sizes:
            for i, s in enumerate(seeds):
                if drop and (rc, sz, s) == drop:
                    continue
                off = offset_for if offset_for == (rc, sz, s) else None
                items = _items(hash((rc, sz, s)) % 10_000, n_items,
                               offset=1000 if off else 0)
                save_run(d / f"{rc}-{sz}-{s}.npz", items,
                         {"recipe": rc, "size": sz, "seed": int(s),
                          "step": 1000, "batch": i, "gain": -1.0 - 0.01 * s})
    return d


def test_a_clean_corpus_becomes_a_population_with_both_phenotypes(tmp_path):
    pop, info = build_population(_corpus(tmp_path), TRAITS)
    assert (pop.N, pop.R, pop.K) == (4, 3, 3)
    assert sorted(pop.phenotypes) == [ACCURACY, MARGIN]
    assert info["n_runs_read"] == 12 and info["dropped_cells"] == []
    assert info["half_A_items"] + info["half_B_items"] == sum(N_ITEMS)
    assert pop.n_items.tolist() == list(N_ITEMS)


def test_recipes_become_the_clusters_and_sizes_the_bands(tmp_path):
    pop, info = build_population(_corpus(tmp_path), TRAITS)
    assert pop.n_clusters == 2
    assert pop.size_bands.tolist() == [0, 1]
    assert info["recipes"] == ["r1", "r2"]
    assert info["sizes"] == ["60M", "90M"]           # ordered by model size
    assert pop.config_ids[0] == ["r1", "60M"]


def test_a_cell_missing_a_run_is_dropped_and_reported(tmp_path):
    d = _corpus(tmp_path, drop=("r1", "60M", 5))
    pop, info = build_population(d, TRAITS)
    assert pop.N == 3
    assert info["dropped_cells"] == [{"cell": ["r1", "60M"], "n_runs": 2}]


def test_runs_are_ordered_by_batch_so_the_index_means_the_same_thing(tmp_path):
    """Otherwise the batch-free contrast points at a different pair per cell."""
    pop, _ = build_population(_corpus(tmp_path), TRAITS)
    assert np.array_equal(pop.batch, np.tile(np.arange(3), (pop.N, 1)))


def test_runs_scored_on_different_items_are_refused(tmp_path):
    d = _corpus(tmp_path)
    save_run(d / "r1-60M-9.npz", _items(3, offset=5000),
             {"recipe": "r1", "size": "60M", "seed": 9, "step": 1000,
              "batch": 3, "gain": -1.0})
    with pytest.raises(ValueError, match="not on one item set"):
        build_population(d, TRAITS)


def test_an_empty_run_directory_says_what_to_run(tmp_path):
    (tmp_path / "runs").mkdir()
    with pytest.raises(FileNotFoundError, match="seednoise build"):
        build_population(tmp_path / "runs", TRAITS)


def test_a_corpus_where_no_cell_is_complete_is_refused(tmp_path):
    d = _corpus(tmp_path, seeds=(2, 4))
    with pytest.raises(ValueError, match="exactly 3 runs"):
        build_population(d, TRAITS, n_runs=3)


def test_the_split_is_reproducible_and_the_halves_are_disjoint(tmp_path):
    d = _corpus(tmp_path)
    a, _ = build_population(d, TRAITS, split_seed=7)
    b, _ = build_population(d, TRAITS, split_seed=7)
    c, _ = build_population(d, TRAITS, split_seed=8)
    assert np.allclose(a.pheno(MARGIN).A, b.pheno(MARGIN).A)
    assert not np.allclose(a.pheno(MARGIN).A, c.pheno(MARGIN).A)


def test_an_explicit_half_mask_overrides_the_drawn_one(tmp_path):
    d = _corpus(tmp_path)
    trait = np.repeat(np.arange(3), N_ITEMS)
    mask = split_items(trait, K=3, seed=99)
    pop, info = build_population(d, TRAITS, half_mask=mask)
    assert info["half_A_items"] == int(mask.sum())


def test_sizes_and_recipes_can_be_restricted(tmp_path):
    d = _corpus(tmp_path)
    pop, info = build_population(d, TRAITS, sizes=["90M"])
    assert pop.N == 2 and info["sizes"] == ["90M"]
    pop, _ = build_population(d, TRAITS, recipes=["r2"])
    assert pop.N == 2 and pop.config_ids[0][0] == "r2"


def test_the_gain_covariate_travels_with_its_run(tmp_path):
    pop, _ = build_population(_corpus(tmp_path), TRAITS)
    assert np.allclose(pop.gainA, pop.gainB)
    assert np.all(pop.gainA < 0)
    assert len(np.unique(pop.gainA[0])) == 3        # one per seed
