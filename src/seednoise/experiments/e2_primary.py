"""E2: the headline estimate, its mediators, its batch-free twin and its intervals."""

from __future__ import annotations

import numpy as np

from seednoise.estimator import correlation, estimate, k_eff, p_flip, sigma_e
from seednoise.inference import (
    cluster_t_interval, config_bootstrap, deff, icc, wild_bootstrap_t,
)
from seednoise.mediation import fit_mediation, residualise
from seednoise.population import ACCURACY, MARGIN
from seednoise.reliability import aggregate_reliability, variance_components

__all__ = ["run_primary"]

MEDIATION_SETS = [(), ("x",), ("s",), ("x", "s")]


def _row(pop, name, which, label, n_boot, seed):
    e = estimate(pop, name, which=which, check=True)
    out = {"label": label, **e.as_row()}
    for fn, key in ((cluster_t_interval, "t"), (wild_bootstrap_t, "wild"),
                    (config_bootstrap, "config")):
        try:
            i = (fn(e.T, e.U, pop.recipe) if key != "config"
                 else fn(e.T, e.U, n_boot=n_boot, seed=seed))
            if key == "wild":
                i = fn(e.T, e.U, pop.recipe, n_boot=n_boot, seed=seed)
        except Exception as exc:                              # noqa: BLE001
            out[f"{key}_error"] = str(exc)
            continue
        out[f"{key}_lo"], out[f"{key}_hi"] = i.lo, i.hi
        out[f"{key}_se"] = i.se
    return out, e


def run_primary(pop, n_boot=4999, seed=0, rho_icc_default=0.15):
    """Every headline number, before and after each mediator, in both phenotypes."""
    rows, mediation, matrices = [], [], {}
    for name in (MARGIN, ACCURACY):
        base_row, base = _row(pop, name, "all", "full contrast set", n_boot, seed)
        rows.append({"phenotype": name, **base_row})
        try:
            free_row, _ = _row(pop, name, "batch_free", "batch-free contrast",
                               n_boot, seed)
            rows.append({"phenotype": name, **free_row})
        except Exception as exc:                              # noqa: BLE001
            rows.append({"phenotype": name, "label": "batch-free contrast",
                         "error": str(exc)})
        for terms in MEDIATION_SETS[1:]:
            fit = fit_mediation(pop, name, terms=terms)
            res = residualise(pop, name, fit)
            label = "after " + " and ".join(
                {"x": "competence", "s": "gain"}[t] for t in terms)
            r, _ = _row(res, list(res.phenotypes)[-1], "all", label, n_boot, seed)
            r["phenotype"] = name
            r["retained_excess"] = ((r["Lambda"] - 1.0) / (base_row["Lambda"] - 1.0)
                                    if base_row["Lambda"] > 1 else float("nan"))
            rows.append(r)
            mediation.extend({"phenotype": name, **m}
                             for m in fit.as_rows(pop.traits))
        S = sigma_e(pop, name)
        comp = variance_components(pop, name)
        matrices[name] = {
            "Sigma_E": S, "R_E": correlation(S), "vbar": comp.vbar,
            "rho_g": aggregate_reliability(base.lambda_hat, comp.vbar),
            "sigma_agg": base.sigma_agg, "K_eff": k_eff(base.lambda_hat, pop.K),
            "icc_T": icc(base.T, pop.recipe),
        }
        matrices[name]["DEFF"] = deff(matrices[name]["icc_T"])

    acc = matrices[ACCURACY]
    deltas = np.array([0.005, 0.01, 0.02, 0.03, 0.05])
    practitioner = [{"delta_accuracy": float(d),
                     "P_flip": float(p_flip(d, acc["sigma_agg"]))} for d in deltas]
    return {"primary": rows, "mediation": mediation, "matrices": matrices,
            "practitioner": practitioner,
            "rho_icc_default": rho_icc_default,
            "DEFF_default": deff(rho_icc_default)}
