"""E4: the pre-committed bake-off and the Cheverud comparison."""

from __future__ import annotations

import numpy as np

from seednoise.baselines import bakeoff
from seednoise.estimator import correlation, phenotypic_correlation, sigma_e
from seednoise.population import ACCURACY, MARGIN

__all__ = ["run_bakeoff"]


def run_bakeoff(pop, n_folds=5, seed=0):
    """Held-out prediction of the aggregate seed SD, plus ``||R_E - R_P||_F``."""
    rows, cheverud = [], []
    for name in (MARGIN, ACCURACY):
        for r in bakeoff(pop, name, n_folds=n_folds, seed=seed):
            rows.append({"phenotype": name, **r})
        RE = correlation(sigma_e(pop, name))
        RP = phenotypic_correlation(pop, name)
        d = np.where(np.isfinite(RE), RE, 0.0) - RP
        off = ~np.eye(pop.K, dtype=bool)
        cheverud.append({
            "phenotype": name,
            "frobenius_RE_minus_RP": float(np.linalg.norm(d, "fro")),
            "mean_offdiag_R_E": float(np.nanmean(RE[off])),
            "mean_offdiag_R_P": float(np.nanmean(RP[off])),
            "corr_of_offdiagonals": float(np.corrcoef(
                RE[off][np.isfinite(RE[off])], RP[off][np.isfinite(RE[off])])[0, 1]),
        })
    best = min((r for r in rows if r["phenotype"] == MARGIN
                and np.isfinite(r["mse_log_sigma_agg"])),
               key=lambda r: r["mse_log_sigma_agg"], default=None)
    return {"bakeoff": rows, "cheverud": cheverud,
            "winner_margin": None if best is None else best["model"]}
