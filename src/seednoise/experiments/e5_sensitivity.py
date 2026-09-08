"""E5: three checks the pre-registration does not run but a reviewer will ask for.

    gain    the run-level multiplicative gain estimated from the data itself, and
            the excess null N5 manufactures at that gain on a simulation whose
            margin levels, seed variances and item noise match the data
    resplit the headline under fresh draws of the A/B item split, which is what
            decides whether a negative cross-half diagonal is a property of the
            battery or of one split
    loo     the headline with each recipe cluster and each size band held out,
            and on each size band alone

None of these changes a registered number; they say how far the registered
numbers move when something arbitrary is varied.
"""

from __future__ import annotations

import numpy as np

from seednoise.estimator import estimate, sigma_e
from seednoise.inference import wild_bootstrap_t
from seednoise.nulls import n5_gain_artifact, summarise
from seednoise.population import ACCURACY, MARGIN, Population
from seednoise.reliability import variance_components
from seednoise.simulate import SimSpec, simulate

__all__ = ["estimate_gain_sd", "matched_spec", "run_gain_calibration",
           "run_resplit", "run_leave_one_out", "GAIN_GRID"]

GAIN_GRID = (0.01, 0.02, 0.03, 0.05, 0.10)


def _lambda_row(pop: Population, name: str, n_boot: int, seed: int) -> dict:
    e = estimate(pop, name, check=False)
    row = {"Lambda": e.lambda_hat, "K_eff": e.k_eff, "rbar_E": e.rbar_e}
    try:
        ci = wild_bootstrap_t(e.T, e.U, pop.recipe, n_boot=n_boot, seed=seed)
        row.update({"wild_lo": ci.lo, "wild_hi": ci.hi, "wild_se": ci.se})
    except Exception as exc:                                  # noqa: BLE001
        row["wild_error"] = str(exc)
    S = sigma_e(pop, name)
    d = np.diag(S)
    row["negative_diagonals"] = int((d <= 0).sum())
    row["min_diagonal"] = float(d.min())
    return row


# -- gain ----------------------------------------------------------------------


def estimate_gain_sd(pop: Population, name: str = MARGIN) -> dict:
    """The across-seed spread of a run-level multiplicative gain, from the data.

    Each run's trait scores are regressed on its configuration's mean profile,
    ``y[c, r, :] = (1 + g[c, r]) ybar[c, :]``, once per half.  The within-
    configuration deviations of ``g`` on the two halves share everything that
    survives the item split, so their cross-half product divided by ``1 - 1/R``
    is free of item noise, the same identity the headline rests on.  Seed
    effects survive the split too, and independent ones leak into the fit with
    variance ``sum_j Sigma_E(j,j) ybar_j^2 / (sum_j ybar_j^2)^2``, which is
    computed from the observed diagonal and subtracted.  What remains is the
    gain plus any seed effect shared across traits in proportion to the mean
    profile, which no population-level statistic can tell from a gain, so the
    number is an upper bound on the sharpness spread and the conservative side
    for gate G5.
    """
    ph = pop.pheno(name)
    S = np.clip(np.diag(sigma_e(pop, name)), 0.0, None)

    def fit(y):
        ybar = y.mean(axis=1, keepdims=True)                  # (N, 1, K)
        den = (ybar * ybar).sum(axis=2)                       # (N, 1)
        g = (y * ybar).sum(axis=2) / den - 1.0
        leak = ((ybar[:, 0, :] ** 2) * S).sum(axis=1) / den[:, 0] ** 2
        return g - g.mean(axis=1, keepdims=True), leak        # (N, R), (N,)

    (dA, leakA), (dB, leakB) = fit(ph.A), fit(ph.B)
    total = float((dA * dB).mean() / (1.0 - 1.0 / pop.R))
    leak = float(0.5 * (leakA.mean() + leakB.mean()))
    var = total - leak
    raw = float(0.5 * ((dA ** 2).mean() + (dB ** 2).mean()) / (1.0 - 1.0 / pop.R))
    return {"gain_sd": float(np.sqrt(var)) if var > 0 else 0.0,
            "gain_var": var, "leak_var": leak, "total_var": total,
            "gain_sd_raw": float(np.sqrt(raw)), "negative": bool(var <= 0)}


def matched_spec(pop: Population, name: str = MARGIN, gain_sd: float = 0.0,
                 seed: int = 0) -> SimSpec:
    """A null population with the data's margin levels, seed variances and item noise.

    ``Sigma_E`` is diagonal, so the truth is ``Lambda = 1`` and everything the
    estimator returns above 1 is manufactured by the gain.  The leverage a gain
    has is the mean margin over the seed standard deviation, trait by trait, and
    that ratio is what this spec copies from the data.
    """
    ph = pop.pheno(name)
    comp = variance_components(pop, name)
    seed_var = np.clip(comp.sigma_e2, 1e-12, None)
    n_items = (tuple([2000] * pop.K) if pop.n_items is None
               else tuple(int(n) for n in np.asarray(pop.n_items).tolist()))
    n_half = np.maximum(np.asarray(n_items, dtype=np.float64) / 2.0, 1.0)
    item_sd = np.sqrt(np.clip(comp.noise2, 1e-12, None) * n_half)
    level = 0.5 * (ph.A.mean(axis=(0, 1)) + ph.B.mean(axis=(0, 1)))
    spec = SimSpec(n_config=pop.N, n_runs=pop.R, traits=tuple(pop.traits),
                   n_items=n_items, item_sd=tuple(item_sd.tolist()),
                   boundary_z=tuple((-level / item_sd).tolist()),
                   gain_sd=float(gain_sd), n_recipes=pop.n_clusters,
                   n_sizes=int(pop.size_bands.size), seed=seed)
    spec.sigma_e = np.diag(seed_var)
    return spec


def run_gain_calibration(pop: Population, grid=GAIN_GRID, n_rep: int = 500,
                         seed: int = 0, matched: bool = True,
                         default: bool = True, include_estimate: bool = True,
                         progress=None) -> dict:
    """N5 across a grid of gains, on the registered spec and on the matched one.

    ``include_estimate=False`` leaves the data-driven gain out of the grid, so
    a long grid can be split over several machines without each of them
    paying for that point again.
    """
    est = estimate_gain_sd(pop, MARGIN)
    lam = {p: estimate(pop, p, check=False).lambda_hat for p in (MARGIN, ACCURACY)}
    points = set(float(g) for g in grid)
    if include_estimate:
        points.add(est["gain_sd"])
    points = sorted(points)
    rows = []
    for g in points:
        if default:
            for r in n5_gain_artifact(gain_sd=g, n_rep=n_rep, n_config=pop.N,
                                      seed=seed):
                rows.append({"spec": "registered", **r})
        if matched:
            got = {MARGIN: [], ACCURACY: []}
            for i in range(n_rep):
                sim = simulate(matched_spec(pop, MARGIN, gain_sd=g,
                                            seed=seed + 104729 * i))
                for p in got:
                    got[p].append(estimate(sim, p, check=False).lambda_hat)
            for p, v in got.items():
                rows.append({"spec": "matched", "null": "N5", "gain_sd": g,
                             "phenotype": p, **summarise(v, 1.0)})
        if progress is not None:
            print(f"  gain_sd={g:.4f} done", flush=True)
    for r in rows:
        r["observed_lambda"] = lam[r["phenotype"]]
        excess = lam[r["phenotype"]] - 1.0
        r["share_of_excess"] = ((r["mean"] - 1.0) / excess if excess > 0
                                else float("nan"))
        r["at_estimated_gain"] = bool(abs(r["gain_sd"] - est["gain_sd"]) < 1e-12)
    return {"gain": rows, "estimate": est}


# -- resplit -------------------------------------------------------------------


def run_resplit(rebuild, seeds, n_boot: int = 4999, seed: int = 0,
                progress=None) -> list:
    """The headline under fresh A/B splits; ``rebuild(split_seed)`` returns a population."""
    rows = []
    for s in seeds:
        pop = rebuild(int(s))
        for p in (MARGIN, ACCURACY):
            r = _lambda_row(pop, p, n_boot, seed)
            S = sigma_e(pop, p)
            for j, t in enumerate(pop.traits):
                r[f"diag_{t}"] = float(S[j, j])
            rows.append({"split_seed": int(s), "phenotype": p, **r})
        if progress is not None:
            print(f"  split {s} done", flush=True)
    return rows


# -- leave one out -------------------------------------------------------------


def run_leave_one_out(pop: Population, recipes=None, sizes=None,
                      n_boot: int = 4999, seed: int = 0) -> list:
    """Lambda with each recipe or size held out, and on each size band alone."""
    recipes = list(recipes) if recipes is not None else \
        [str(i) for i in range(pop.n_clusters)]
    sizes = list(sizes) if sizes is not None else \
        [str(i) for i in range(int(pop.size_bands.size))]
    rows = []
    plans = []
    for i, r in enumerate(recipes):
        plans.append(("drop_recipe", r, np.flatnonzero(pop.recipe != i)))
    for i, z in enumerate(sizes):
        plans.append(("drop_size", z, np.flatnonzero(pop.size != i)))
        plans.append(("only_size", z, np.flatnonzero(pop.size == i)))
    for kind, label, idx in plans:
        sub = pop.subset(idx)
        for p in (MARGIN, ACCURACY):
            rows.append({"kind": kind, "held": label, "N": int(sub.N),
                         "n_clusters": int(sub.n_clusters), "phenotype": p,
                         **_lambda_row(sub, p, n_boot, seed)})
    return rows
