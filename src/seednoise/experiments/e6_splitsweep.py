"""E6: every estimation-recipe subset the registration could have drawn.

The registration described eight screening recipes and 17 estimation recipes,
with the headline computed on the 85 configurations of the estimation set.  No
code ever drew that partition: every table under ``results/`` uses all 25 recipes
and 125 configurations.  This module reports what the headline would have been
under any of the C(25, 17) = 1081575 estimation subsets, which is cheap because
each configuration's ``T_c`` and ``U_c`` do not depend on which other
configurations are present, so a subset's ``Lambda`` is a ratio of two sums.

Everything here is post hoc and is labelled as such in the tables it writes.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

from seednoise.estimator import _basis_for, _tu, k_eff
from seednoise.halves import MASTER_SEED
from seednoise.population import ACCURACY, MARGIN, Population

__all__ = ["recipe_tu", "sweep_lambda", "seed_partition", "run_splitsweep",
           "REGISTERED_THRESHOLDS"]

# The registered predictions, in Lambda units: rbar_E 0.091 on the margin gives
# sqrt(1 + 9 x 0.091) = 1.349 at K = 10, and the accuracy prediction was 1.40.
REGISTERED_THRESHOLDS = {MARGIN: 1.349, ACCURACY: 1.40}
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def recipe_tu(pop: Population, name: str, which: str = "all"):
    """Per-recipe sums of ``T_c`` and ``U_c`` and the configuration count.

    Returns ``(codes, T, U, n)`` with one entry per recipe code in ``pop.recipe``,
    sorted by code, so ``sqrt(T.sum() / U.sum())`` is the full-population estimate.
    """
    tu = _tu(pop, name, _basis_for(pop, which))
    codes = np.unique(pop.recipe)
    T = np.array([tu.T[pop.recipe == c].sum() for c in codes])
    U = np.array([tu.U[pop.recipe == c].sum() for c in codes])
    n = np.array([int((pop.recipe == c).sum()) for c in codes])
    return codes, T, U, n


def sweep_lambda(T: np.ndarray, U: np.ndarray, n_estimation: int,
                 chunk: int = 200_000):
    """``Lambda`` for every ``n_estimation``-subset of the recipes, in the order
    ``itertools.combinations`` yields them, plus the index arrays of the
    subsets that gave the smallest and largest values."""
    R = len(T)
    if not 1 <= n_estimation <= R:
        raise ValueError(f"n_estimation must be in 1..{R}, got {n_estimation}")
    total = comb(R, n_estimation)
    lam = np.empty(total)
    lo = hi = None
    it = combinations(range(R), n_estimation)
    done = 0
    while done < total:
        idx = np.fromiter(it, dtype=np.dtype((np.int64, n_estimation)),
                          count=min(chunk, total - done))
        idx = idx.reshape(-1, n_estimation)
        ratio = T[idx].sum(axis=1) / U[idx].sum(axis=1)
        # A negative ratio has no real root and is reported as nan, as in
        # ``TU.lambda_hat``; nan never wins the argmin or argmax below.
        out = np.where(ratio >= 0, np.sqrt(np.abs(ratio)), np.nan)
        lam[done: done + len(out)] = out
        if np.isfinite(out).any():
            a, b = np.nanargmin(out), np.nanargmax(out)
            if lo is None or out[a] < lam[lo[0]]:
                lo = (done + a, idx[a])
            if hi is None or out[b] > lam[hi[0]]:
                hi = (done + b, idx[b])
        done += len(out)
    if lo is None:
        raise ValueError("every subset gave a negative T/U ratio; nothing to sweep")
    return lam, lo[1], hi[1]


def seed_partition(recipe_names, n_estimation: int, seed: int = MASTER_SEED):
    """The partition the registered seed would have drawn, had it been drawn.

    Sorted names permuted by ``numpy.random.default_rng(seed)``; the last
    ``n_estimation`` are the estimation set.  This is a post hoc reconstruction
    of a rule the registration did not spell out, not a recovered artefact.
    """
    names = sorted(str(r) for r in recipe_names)
    perm = np.random.default_rng(seed).permutation(len(names))
    est = sorted(names[i] for i in perm[len(names) - n_estimation:])
    scr = sorted(names[i] for i in perm[: len(names) - n_estimation])
    return scr, est


def run_splitsweep(pop: Population, recipe_names=None, n_estimation: int = 17,
                   thresholds=None, seed: int = MASTER_SEED, which: str = "all"):
    """The sweep on both phenotypes, as the rows ``tab_splitsweep`` prints."""
    thresholds = dict(REGISTERED_THRESHOLDS if thresholds is None else thresholds)
    codes = np.unique(pop.recipe)
    if recipe_names is None:
        recipe_names = [f"recipe-{int(c)}" for c in codes]
    recipe_names = [str(r) for r in recipe_names]
    if len(recipe_names) != len(codes):
        raise ValueError(f"{len(recipe_names)} recipe names for {len(codes)} "
                         "recipe codes; pass the names in code order")
    scr, est = seed_partition(recipe_names, n_estimation, seed)
    est_codes = np.array([recipe_names.index(r) for r in est])

    rows, per_recipe, extremes = [], {}, {}
    for name in (MARGIN, ACCURACY):
        _, T, U, n = recipe_tu(pop, name, which)
        lam, lo, hi = sweep_lambda(T, U, n_estimation)
        finite = lam[np.isfinite(lam)]
        thr = thresholds.get(name)
        row = {
            "phenotype": name, "contrast": which,
            "n_recipes": int(len(codes)), "n_estimation": int(n_estimation),
            "n_subsets": int(lam.size), "n_subsets_finite": int(finite.size),
            "Lambda_all_recipes": float(np.sqrt(T.sum() / U.sum())),
            "Lambda_seed_partition": float(np.sqrt(T[est_codes].sum()
                                                   / U[est_codes].sum())),
            "min": float(finite.min()), "max": float(finite.max()),
        }
        for q in QUANTILES:
            row[f"q{int(round(q * 100)):02d}"] = float(np.quantile(finite, q))
        row["K_eff_min"] = k_eff(row["max"], pop.K)
        row["K_eff_max"] = k_eff(row["min"], pop.K)
        row["threshold"] = float("nan") if thr is None else float(thr)
        row["share_above_threshold"] = (float("nan") if thr is None
                                        else float((finite > thr).mean()))
        row["share_below_one"] = float((finite < 1.0).mean())
        rows.append(row)
        per_recipe[name] = (T, U, n)
        extremes[name] = {"min": [recipe_names[i] for i in lo],
                          "max": [recipe_names[i] for i in hi]}

    recipe_rows = []
    for i, r in enumerate(recipe_names):
        rr = {"recipe": r, "n_config": int(per_recipe[MARGIN][2][i]),
              "in_seed_estimation_set": r in est}
        for name, (T, U, _) in per_recipe.items():
            rr[f"T_{name}"] = float(T[i])
            rr[f"U_{name}"] = float(U[i])
        recipe_rows.append(rr)

    source = {
        "note": ("post hoc: the registered 8/17 screening-estimation partition "
                 "was never implemented and no registered partition exists; "
                 "every registered table uses all recipes"),
        "n_recipes": int(len(codes)), "n_estimation": int(n_estimation),
        "n_subsets": int(comb(len(codes), n_estimation)),
        "thresholds": {k: float(v) for k, v in thresholds.items()},
        "seed": int(seed), "seed_partition": {"screening": scr, "estimation": est},
        "extreme_subsets": extremes,
    }
    return {"splitsweep": rows, "recipes": recipe_rows, "source": source}
