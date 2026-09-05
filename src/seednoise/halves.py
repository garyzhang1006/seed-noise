"""The frozen item split, and the clustered split that gate G3 falls back to.

The A/B split is drawn once from a master seed and reused by every run, every
configuration and both arms, because the cross-half identity needs half A and
half B to be the same item sets throughout; a split redrawn per run would make
the two halves' sampling noise correlated through the shared draw.

Independence of the two halves' item noise is the one assumption the identity
cannot do without.  Items that share a passage violate it, since a run that
happens to parse one BoolQ passage well is right about every question on that
passage, so the default split is clustered on the passage key wherever one
exists and falls back to items only where it does not.
"""

from __future__ import annotations

import numpy as np

__all__ = ["split_items", "split_clustered", "half_sizes", "MASTER_SEED"]

# The one number that makes the split reproducible from the repository alone.
MASTER_SEED = 20260101


def split_items(trait: np.ndarray, K: int, seed: int = MASTER_SEED) -> np.ndarray:
    """Balanced random half per trait; returns the boolean mask of half A.

    Splitting within trait rather than across the pooled battery keeps both halves
    covering all ten traits, which ``half_scores`` requires and which a pooled
    split would violate by chance for the 500-item benchmarks.
    """
    trait = np.asarray(trait)
    rng = np.random.default_rng(seed)
    mask = np.zeros(trait.shape[0], dtype=bool)
    for j in range(K):
        idx = np.flatnonzero(trait == j)
        if idx.size < 2:
            raise ValueError(
                f"trait {j} has {idx.size} items, which cannot be split into two "
                "non-empty halves"
            )
        take = rng.permutation(idx)[: idx.size // 2]
        mask[take] = True
    return mask


def split_clustered(trait: np.ndarray, cluster: np.ndarray, K: int,
                    seed: int = MASTER_SEED) -> np.ndarray:
    """Half split that keeps every item sharing a passage on the same side.

    ``cluster`` is any per-item grouping key; items with a unique key behave
    exactly as under ``split_items``.  Clusters are assigned greedily to whichever
    half currently holds fewer items of that trait, which keeps the halves close
    to equal even when cluster sizes are very uneven.
    """
    trait = np.asarray(trait)
    cluster = np.asarray(cluster)
    if cluster.shape != trait.shape:
        raise ValueError(f"cluster {cluster.shape} and trait {trait.shape} disagree")
    rng = np.random.default_rng(seed)
    mask = np.zeros(trait.shape[0], dtype=bool)
    for j in range(K):
        idx = np.flatnonzero(trait == j)
        if idx.size < 2:
            raise ValueError(
                f"trait {j} has {idx.size} items, which cannot be split into two "
                "non-empty halves"
            )
        keys, inv = np.unique(cluster[idx], return_inverse=True)
        if keys.size < 2:
            raise ValueError(
                f"trait {j} has every item in one cluster, so a clustered split "
                "would leave one half empty; use split_items for this trait"
            )
        sizes = np.bincount(inv)
        # Largest clusters first, so the greedy balance is not spoiled at the end.
        order = np.argsort(-sizes + rng.random(sizes.size) * 1e-6)
        load = [0, 0]
        side = np.zeros(keys.size, dtype=bool)
        for c in order:
            to_a = load[0] <= load[1]
            side[c] = to_a
            load[0 if to_a else 1] += int(sizes[c])
        mask[idx] = side[inv]
    return mask


def half_sizes(trait: np.ndarray, mask: np.ndarray, K: int) -> np.ndarray:
    """``(K, 2)`` item counts per trait per half, which every gate reports."""
    trait, mask = np.asarray(trait), np.asarray(mask, dtype=bool)
    a = np.bincount(trait[mask], minlength=K)[:K]
    b = np.bincount(trait[~mask], minlength=K)[:K]
    return np.stack([a, b], axis=1)
