"""The A/B split: balanced, deterministic, and cluster-respecting when it must be."""

from __future__ import annotations

import numpy as np

from seednoise.halves import MASTER_SEED, half_sizes, split_clustered, split_items


def _trait(counts):
    return np.repeat(np.arange(len(counts)), counts)


def test_every_trait_is_split_within_one_item():
    trait = _trait([1172, 2376, 3270, 1221, 501])
    mask = split_items(trait, K=5)
    for j in range(5):
        n = int((trait == j).sum())
        assert abs(int(mask[trait == j].sum()) - n // 2) <= 1


def test_an_odd_trait_leaves_the_odd_item_on_one_side_only():
    trait = _trait([7])
    sizes = half_sizes(trait, split_items(trait, K=1), K=1)
    assert sizes.sum() == 7 and abs(sizes[0, 0] - sizes[0, 1]) == 1


def test_the_split_is_a_deterministic_function_of_the_seed():
    trait = _trait([100, 100])
    a = split_items(trait, K=2, seed=MASTER_SEED)
    b = split_items(trait, K=2, seed=MASTER_SEED)
    c = split_items(trait, K=2, seed=MASTER_SEED + 1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_a_clustered_split_never_puts_one_cluster_in_both_halves():
    """Items sharing a passage leak across halves unless the passage does not split."""
    trait = np.zeros(60, dtype=np.int64)
    cluster = np.repeat(np.arange(12), 5)
    mask = split_clustered(trait, cluster, K=1)
    for c in np.unique(cluster):
        assert len(set(mask[cluster == c].tolist())) == 1


def test_a_clustered_split_still_balances_the_halves_approximately():
    trait = np.zeros(120, dtype=np.int64)
    cluster = np.repeat(np.arange(24), 5)
    mask = split_clustered(trait, cluster, K=1)
    assert abs(int(mask.sum()) - 60) <= 5


def test_half_sizes_reports_both_sides_per_trait():
    trait = _trait([10, 20])
    sizes = half_sizes(trait, split_items(trait, K=2), K=2)
    assert sizes.shape == (2, 2)
    assert sizes[0].sum() == 10 and sizes[1].sum() == 20
