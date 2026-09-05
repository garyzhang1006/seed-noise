r"""The two co-primary phenotypes, from per-choice log-likelihoods to trait scores.

For item ``i`` with gold answer ``g`` and answer choices ``k``,

    m_i = l(g)/bytes(g) - max_{k != g} l(k)/bytes(k),      a_i = 1[m_i > 0]

so accuracy is the thresholded image of the margin and the two are computed from
one pass over the same records.  Byte normalisation rather than token
normalisation is what makes the margin comparable across tokenizers, and it is
also what makes the leading space of a continuation matter: a choice scored with
its leading space carries one more byte than the same choice scored without it,
which moves the per-byte score of every choice by a different amount and changes
which choice wins.  The convention is therefore fixed once, recorded on the
output, and checked rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "per_byte", "reduce_choices", "ItemPhenotypes", "half_scores",
    "count_bytes", "MARGIN_EPS",
]

# Ties are resolved against the model: a margin of exactly zero scores as wrong,
# which matters only for degenerate duplicate choices but should never be silent.
MARGIN_EPS = 0.0


def count_bytes(text: str, leading_space: bool = True) -> int:
    """Bytes of a continuation under the fixed convention.

    ``leading_space`` records whether the space that separates the context from
    the continuation is scored as part of the continuation.  Both conventions are
    defensible and neither is reversible after the fact, so the choice travels
    with the data.
    """
    s = text if leading_space else text.lstrip(" ")
    n = len(s.encode("utf-8"))
    if n == 0:
        raise ValueError(
            f"continuation {text!r} is zero bytes under leading_space="
            f"{leading_space}, so its per-byte score is undefined"
        )
    return n


def per_byte(logprob, n_bytes) -> np.ndarray:
    """``l(k)/bytes(k)``, refusing the division that would quietly return inf."""
    lp = np.asarray(logprob, dtype=np.float64)
    nb = np.asarray(n_bytes, dtype=np.float64)
    if lp.shape != nb.shape:
        raise ValueError(f"logprob {lp.shape} and n_bytes {nb.shape} disagree")
    bad = ~(nb > 0)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} of {nb.size} choices have non-positive byte counts, "
            f"first at index {int(np.flatnonzero(bad)[0])}"
        )
    if not np.isfinite(lp).all():
        raise ValueError(
            f"{int((~np.isfinite(lp)).sum())} log-likelihoods are not finite; a "
            "truncated or failed scoring pass must not be reduced to a phenotype"
        )
    return lp / nb


@dataclass(frozen=True)
class ItemPhenotypes:
    """Per-item margin and correctness for one run, aligned to ``item_id``."""

    item_id: np.ndarray          # (n_items,) sorted, unique
    trait: np.ndarray            # (n_items,) trait index
    margin: np.ndarray           # (n_items,) float
    correct: np.ndarray          # (n_items,) bool
    group: np.ndarray | None = None   # (n_items,) sub-trait for macro-averaging

    def __post_init__(self):
        n = self.item_id.shape[0]
        if self.group is None:
            # One group per trait reduces the macro-average to a plain item mean.
            object.__setattr__(self, "group", self.trait.copy())
        for nm in ("trait", "margin", "correct", "group"):
            v = getattr(self, nm)
            if v.shape != (n,):
                raise ValueError(f"{nm} is {v.shape}, expected ({n},)")
        if not np.isfinite(self.margin).all():
            raise ValueError("margins contain non-finite values")

    @property
    def n_items(self) -> int:
        return int(self.item_id.shape[0])


def reduce_choices(item_id, trait, score, is_gold, group=None) -> ItemPhenotypes:
    """Collapse per-choice scores to one margin and one correctness per item.

    ``score`` is already per-byte.  Exactly one gold per item and at least two
    choices per item are required, because an item with a single choice has no
    margin and an item with no gold cannot be scored, and silently dropping
    either would change the item set between runs and break the cross-half
    identity, which assumes every run is scored on the same items.
    """
    item_id = np.asarray(item_id)
    trait = np.asarray(trait)
    score = np.asarray(score, dtype=np.float64)
    is_gold = np.asarray(is_gold, dtype=bool)
    shapes = {item_id.shape, trait.shape, score.shape, is_gold.shape}
    if len(shapes) != 1 or item_id.ndim != 1:
        raise ValueError(f"inputs must be one-dimensional and aligned, got {shapes}")
    if item_id.size == 0:
        raise ValueError("no choices to reduce")

    order = np.argsort(item_id, kind="stable")
    iid, tr, sc, gold = item_id[order], trait[order], score[order], is_gold[order]
    uniq, start, counts = np.unique(iid, return_index=True, return_counts=True)

    if (counts < 2).any():
        bad = uniq[counts < 2][:3]
        raise ValueError(
            f"{int((counts < 2).sum())} items have fewer than two answer choices, "
            f"first {bad.tolist()}; such an item has no margin"
        )
    n_gold = np.add.reduceat(gold.astype(np.int64), start)
    if not np.all(n_gold == 1):
        bad = uniq[n_gold != 1][:3]
        raise ValueError(
            f"{int((n_gold != 1).sum())} items do not have exactly one gold choice, "
            f"first {bad.tolist()} with {n_gold[n_gold != 1][:3].tolist()} golds"
        )
    trait_span = np.array([np.unique(tr[s:s + c]).size
                           for s, c in zip(start, counts)])
    if (trait_span != 1).any():
        raise ValueError(
            f"{int((trait_span != 1).sum())} items carry choices from more than one "
            "trait, so the item ids are not unique across tasks"
        )

    gold_score = sc[gold]                       # one per item, in item order
    masked = np.where(gold, -np.inf, sc)
    best_other = np.maximum.reduceat(masked, start)
    margin = gold_score - best_other
    return ItemPhenotypes(
        item_id=uniq, trait=tr[start], margin=margin,
        correct=margin > MARGIN_EPS,
        group=None if group is None else np.asarray(group)[order][start],
    )


def half_scores(items: ItemPhenotypes, half: np.ndarray, K: int):
    """``(K,)`` trait means of margin and accuracy over the items in one half.

    Each trait is the mean over its groups of the group's item mean, which makes
    MMLU the macro-average over its 57 subjects that the benchmark is normally
    reported as, and leaves every single-group trait a plain item mean.

    A trait or group with no items in the half is an error rather than a ``nan``:
    the estimator multiplies half A by half B trait by trait, and a missing group
    on one side would silently reweight that trait for that run alone.
    """
    half = np.asarray(half, dtype=bool)
    if half.shape != (items.n_items,):
        raise ValueError(f"half mask is {half.shape} for {items.n_items} items")
    tr, gr = items.trait[half], items.group[half]
    counts = np.bincount(tr, minlength=K)
    if (counts == 0).any():
        raise ValueError(
            f"traits {np.flatnonzero(counts == 0).tolist()} have no items in this "
            "half, so their half score is undefined"
        )
    keys, inv = np.unique(gr, return_inverse=True)
    n_g = np.bincount(inv, minlength=keys.size)
    trait_of_group = np.zeros(keys.size, dtype=np.int64)
    trait_of_group[inv] = tr
    out = []
    for vals in (items.margin[half], items.correct[half].astype(np.float64)):
        gm = np.bincount(inv, weights=vals, minlength=keys.size) / n_g
        n_t = np.bincount(trait_of_group, minlength=K)[:K]
        out.append(np.bincount(trait_of_group, weights=gm, minlength=K)[:K] / n_t)
    return out[0], out[1]
