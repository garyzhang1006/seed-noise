"""E1: the coverage, information, symmetry and held-out-loss gates.

The registration described eight screening recipes that would carry every design
decision and 17 estimation recipes opened only once these gates had fired.  That
partition was never implemented: nothing in this package draws it, no frozen
plan or hash exists, and every table under ``results/`` runs the gates and the
headline on all 25 recipes and 125 configurations.  ``seednoise splitsweep``
(E6) reports the range the headline takes over every 17-recipe subset, so the
paper can say what the registered design would have shown.
"""

from __future__ import annotations

from seednoise.estimator import correlation, estimate, sigma_e
from seednoise.gates import (
    g0_coverage, g2_information_ratio, g3_cross_half_symmetry, g7_heldout_loss,
    gate_table,
)
from seednoise.population import ACCURACY, MARGIN
from seednoise.reliability import variance_components

__all__ = ["run_screen"]


def run_screen(pop, n_cells_parsed=125, n_estimation=85, has_heldout_loss=False):
    """Gates G0, G2, G3 and G7, plus the reliability table they rest on.

    ``n_estimation`` is the number of configurations the headline is computed
    on; the command line passes ``pop.N``, which is 125 on the release, because
    no screening set is held out.
    """
    comp = {p: variance_components(pop, p) for p in (MARGIN, ACCURACY)}
    acc_by_trait = 0.5 * (pop.pheno(ACCURACY).A.mean(axis=(0, 1))
                          + pop.pheno(ACCURACY).B.mean(axis=(0, 1)))
    gates = [
        g0_coverage(n_cells_parsed, 125, n_estimation),
        g2_information_ratio(acc_by_trait),
        g3_cross_half_symmetry(pop, MARGIN),
        g7_heldout_loss(has_heldout_loss),
    ]
    rel_rows = []
    for p, c in comp.items():
        for row in c.as_rows():
            rel_rows.append({"phenotype": p, **row})
    return {
        "gates": gate_table(gates),
        "reliability": rel_rows,
        "screen_lambda": [estimate(pop, p, check=True).as_row()
                          for p in (MARGIN, ACCURACY)],
        "accuracy_by_trait": {t: float(a) for t, a in zip(pop.traits, acc_by_trait)},
        "R_E_margin": correlation(sigma_e(pop, MARGIN)),
        "vbar": {p: comp[p].vbar for p in comp},
    }
