"""From per-choice log-likelihoods to trait scores, including the ways it goes wrong.

The properties tested here are the ones the estimator silently assumes: the same
items in every run, one gold per item, and a macro-average for MMLU.
"""

from __future__ import annotations

import numpy as np
import pytest

from seednoise.phenotypes import (
    ItemPhenotypes, count_bytes, half_scores, per_byte, reduce_choices,
)


def _choices(n_items=6, n_choices=4, K=2, seed=0, group=None):
    rng = np.random.default_rng(seed)
    iid = np.repeat(np.arange(n_items), n_choices)
    trait = iid % K
    gold = np.zeros(iid.size, dtype=bool)
    gold[::n_choices] = True
    score = rng.standard_normal(iid.size)
    g = None if group is None else np.repeat(group, n_choices)
    return iid, trait, score, gold, g


def test_margin_is_the_gold_minus_the_best_rival():
    iid = np.array([0, 0, 0, 1, 1, 1])
    trait = np.zeros(6, dtype=int)
    score = np.array([-1.0, -2.0, -3.0, -5.0, -4.0, -9.0])
    gold = np.array([True, False, False, True, False, False])
    ph = reduce_choices(iid, trait, score, gold)
    assert ph.margin == pytest.approx([1.0, -1.0])
    assert ph.correct.tolist() == [True, False]


def test_an_exact_tie_scores_as_wrong():
    """A zero margin is a coin flip the model did not win, and must not float."""
    ph = reduce_choices([0, 0], [0, 0], [-1.0, -1.0], [True, False])
    assert ph.margin[0] == 0.0
    assert not ph.correct[0]


def test_items_come_back_sorted_and_unique_whatever_the_input_order():
    iid, trait, score, gold, _ = _choices(seed=1)
    order = np.random.default_rng(0).permutation(iid.size)
    a = reduce_choices(iid, trait, score, gold)
    b = reduce_choices(iid[order], trait[order], score[order], gold[order])
    assert np.array_equal(a.item_id, b.item_id)
    assert a.margin == pytest.approx(b.margin)
    assert np.array_equal(a.item_id, np.unique(a.item_id))


def test_a_single_choice_item_is_refused():
    with pytest.raises(ValueError, match="fewer than two"):
        reduce_choices([0, 1, 1], [0, 0, 0], [-1.0, -1.0, -2.0],
                       [True, True, False])


def test_two_golds_or_no_gold_are_refused():
    with pytest.raises(ValueError, match="exactly one gold"):
        reduce_choices([0, 0], [0, 0], [-1.0, -2.0], [True, True])
    with pytest.raises(ValueError, match="exactly one gold"):
        reduce_choices([0, 0], [0, 0], [-1.0, -2.0], [False, False])


def test_an_item_id_reused_across_traits_is_refused():
    """Colliding ids would merge two different questions into one phenotype."""
    with pytest.raises(ValueError, match="more than one"):
        reduce_choices([0, 0], [0, 1], [-1.0, -2.0], [True, False])


def test_per_byte_refuses_zero_bytes_and_non_finite_scores():
    with pytest.raises(ValueError, match="non-positive byte counts"):
        per_byte([-1.0, -2.0], [3.0, 0.0])
    with pytest.raises(ValueError, match="not finite"):
        per_byte([-1.0, -np.inf], [3.0, 4.0])


def test_the_leading_space_convention_changes_the_byte_count():
    assert count_bytes(" Paris") == count_bytes("Paris") + 1
    assert count_bytes(" Paris", leading_space=False) == count_bytes("Paris")
    with pytest.raises(ValueError, match="zero bytes"):
        count_bytes(" ", leading_space=False)


def test_byte_counts_are_utf8_not_characters():
    assert count_bytes("é", leading_space=False) == 2


def test_half_scores_are_plain_item_means_without_groups():
    iid = np.arange(4)
    trait = np.array([0, 0, 1, 1])
    ph = ItemPhenotypes(iid, trait, np.array([1.0, 3.0, -1.0, 5.0]),
                        np.array([True, True, False, True]))
    m, a = half_scores(ph, np.ones(4, dtype=bool), K=2)
    assert m == pytest.approx([2.0, 2.0])
    assert a == pytest.approx([1.0, 0.5])


def test_a_grouped_trait_is_macro_averaged_over_its_groups():
    """One large subject must not outvote a small one, which is what MMLU needs."""
    iid = np.arange(5)
    trait = np.zeros(5, dtype=np.int64)
    group = np.array([0, 0, 0, 0, 1])          # four items, then one
    ph = ItemPhenotypes(iid, trait, np.array([1.0, 1.0, 1.0, 1.0, 5.0]),
                        np.ones(5, dtype=bool), group=group)
    m, _ = half_scores(ph, np.ones(5, dtype=bool), K=1)
    assert m == pytest.approx([3.0])            # not the item mean of 1.8
    flat = ItemPhenotypes(iid, trait, ph.margin, ph.correct)
    assert half_scores(flat, np.ones(5, dtype=bool), K=1)[0] == pytest.approx([1.8])


def test_a_trait_with_no_items_in_a_half_is_an_error_not_a_nan():
    iid = np.arange(4)
    ph = ItemPhenotypes(iid, np.array([0, 0, 1, 1]), np.zeros(4),
                        np.ones(4, dtype=bool))
    mask = np.array([True, True, False, False])
    with pytest.raises(ValueError, match="no items in this"):
        half_scores(ph, mask, K=2)


def test_a_mask_of_the_wrong_length_is_refused():
    ph = ItemPhenotypes(np.arange(3), np.zeros(3, dtype=int), np.zeros(3),
                        np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="half mask is"):
        half_scores(ph, np.ones(4, dtype=bool), K=1)


def test_non_finite_margins_cannot_be_stored_in_a_phenotype():
    with pytest.raises(ValueError, match="non-finite"):
        ItemPhenotypes(np.arange(2), np.zeros(2, dtype=int),
                       np.array([0.0, np.nan]), np.ones(2, dtype=bool))


def test_accuracy_is_the_threshold_of_the_margin_on_real_reductions():
    iid, trait, score, gold, _ = _choices(n_items=50, seed=3)
    ph = reduce_choices(iid, trait, score, gold)
    assert np.array_equal(ph.correct, ph.margin > 0)
