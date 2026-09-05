"""Assemble the rectangular population from the reduced runs on disk.

Everything upstream is per-run and item-level; everything downstream is the
``(N, R, K)`` array the estimator reads.  This is the only place the two meet, so
it is also the only place that can silently misalign them, and every alignment
assumption is checked here rather than assumed: the item ids must be identical
across runs, every configuration must carry the same number of runs, and the
half split must be the same one for every run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from seednoise.halves import MASTER_SEED, split_items
from seednoise.phenotypes import half_scores
from seednoise.population import ACCURACY, MARGIN, Phenotype, Population
from seednoise.store import load_run

__all__ = ["build_population", "SIZE_ORDER"]

SIZE_ORDER = ["4M", "6M", "8M", "10M", "14M", "16M", "20M", "60M", "90M",
              "150M", "300M", "530M", "750M", "1B"]


def build_population(run_dir, traits, n_runs=3, split_seed=MASTER_SEED,
                     half_mask=None, sizes=None, recipes=None):
    """Read every reduced run in ``run_dir`` and return the population.

    Configurations with a different number of runs than ``n_runs`` are dropped and
    reported rather than padded, because a padded cell would enter the estimator
    with a run that does not exist.
    """
    files = sorted(Path(run_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(
            f"no reduced runs in {run_dir}; run 'seednoise build' first"
        )
    K = len(traits)
    runs, ref_ids = [], None
    for f in files:
        items, meta = load_run(f)
        if ref_ids is None:
            ref_ids = items.item_id
        elif not np.array_equal(items.item_id, ref_ids):
            raise ValueError(
                f"{f.name} covers {items.n_items} items that do not match the "
                f"{ref_ids.size} of the first run; the runs are not on one item set"
            )
        runs.append((items, meta))

    if half_mask is None:
        half_mask = split_items(runs[0][0].trait, K, seed=split_seed)
    half_mask = np.asarray(half_mask, dtype=bool)

    cells: dict = {}
    for items, meta in runs:
        if sizes is not None and meta["size"] not in sizes:
            continue
        if recipes is not None and meta["recipe"] not in recipes:
            continue
        mA, aA = half_scores(items, half_mask, K)
        mB, aB = half_scores(items, ~half_mask, K)
        cells.setdefault((meta["recipe"], meta["size"]), []).append(
            (meta, mA, aA, mB, aB))

    keep = {k: v for k, v in cells.items() if len(v) == n_runs}
    dropped = [{"cell": list(k), "n_runs": len(v)}
               for k, v in cells.items() if len(v) != n_runs]
    if not keep:
        raise ValueError(
            f"no configuration has exactly {n_runs} runs; found run counts "
            f"{sorted({len(v) for v in cells.values()})}"
        )

    order = sorted(keep)
    N = len(order)
    mA = np.zeros((N, n_runs, K)); aA = np.zeros((N, n_runs, K))
    mB = np.zeros((N, n_runs, K)); aB = np.zeros((N, n_runs, K))
    gA = np.zeros((N, n_runs)); gB = np.zeros((N, n_runs))
    batch = np.zeros((N, n_runs), dtype=np.int64)
    # The clustering and the size band are integer codes downstream, so the names
    # are mapped once here and the map travels in the info dictionary; the pair
    # itself stays on the configuration as its id.
    recipe_names = sorted({k[0] for k in order})
    size_names = sorted({k[1] for k in order},
                        key=lambda z: SIZE_ORDER.index(z) if z in SIZE_ORDER else 999)
    recipe_code = {r: i for i, r in enumerate(recipe_names)}
    size_code = {z: i for i, z in enumerate(size_names)}
    recipe = np.zeros(N, dtype=np.int64)
    size = np.zeros(N, dtype=np.int64)
    for c, key in enumerate(order):
        # Batch order, not file order, so run r means the same batch in every cell.
        rows = sorted(keep[key], key=lambda t: t[0]["batch"])
        recipe[c], size[c] = recipe_code[key[0]], size_code[key[1]]
        for r, (meta, m_a, a_a, m_b, a_b) in enumerate(rows):
            mA[c, r], aA[c, r], mB[c, r], aB[c, r] = m_a, a_a, m_b, a_b
            batch[c, r] = meta["batch"]
            gA[c, r] = gB[c, r] = meta["gain"]
    n_items = np.bincount(runs[0][0].trait, minlength=K)[:K]
    pop = Population(
        {MARGIN: Phenotype(MARGIN, mA, mB), ACCURACY: Phenotype(ACCURACY, aA, aB)},
        gainA=gA, gainB=gB, batch=batch, recipe=recipe, size=size,
        traits=list(traits), config_ids=[list(k) for k in order], n_items=n_items,
    )
    return pop, {"n_config": N, "n_runs_read": len(runs), "dropped_cells": dropped,
                 "recipes": recipe_names, "sizes": size_names,
                 "half_A_items": int(half_mask.sum()),
                 "half_B_items": int((~half_mask).sum()),
                 "split_seed": int(split_seed)}
