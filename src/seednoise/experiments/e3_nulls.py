"""E3: the six nulls, and the two gates that read them."""

from __future__ import annotations

from seednoise.estimator import estimate
from seednoise.gates import g1_accuracy_power, g4_calibration, g5_sharpness, gate_table
from seednoise.nulls import (
    n1_calibration, n2_permutation, n4_zero_mediation, n5_gain_artifact,
    n6_batch_offset,
)
from seednoise.population import ACCURACY, MARGIN
from seednoise.reliability import aggregate_reliability, variance_components

__all__ = ["run_nulls"]


def run_nulls(pop, n_rep=2000, n_rep_slow=500, rbars=None, gain_sd=0.05,
              offset_sd=0.05, seed=0, progress=None):
    """Run every null and return the rows the nulls table and gates G4, G5 need."""
    kw = dict(n_config=pop.N, seed=seed)
    rbars = rbars or (0.0, 0.05, 0.10, 0.20, 0.30, 0.50)
    n1 = n1_calibration(rbars=rbars, n_rep=n_rep, which=("all", "batch_free"),
                        progress=progress, **kw)
    n2 = [n2_permutation(pop, p, n_rep=min(n_rep, 500), seed=seed)
          for p in (MARGIN, ACCURACY)]
    n4 = [n4_zero_mediation(pop, p, n_rep=n_rep_slow, seed=seed)
          for p in (MARGIN, ACCURACY)]
    n5 = n5_gain_artifact(gain_sd=gain_sd, n_rep=n_rep, **kw)
    n6 = n6_batch_offset(offset_sd=offset_sd, n_rep=n_rep, **kw)

    null_sd = {}
    for p in (MARGIN, ACCURACY):
        z = [r for r in n1 if r["rbar_E_true"] == 0.0
             and r["phenotype"] == p and r["contrast"] == "all"]
        null_sd[p] = z[0]["sd"] if z else float("nan")

    comp = variance_components(pop, ACCURACY)
    lam_m = estimate(pop, MARGIN, check=False).lambda_hat
    gates = [
        g1_accuracy_power(null_sd[ACCURACY],
                          aggregate_reliability(
                              estimate(pop, ACCURACY, check=False).lambda_hat,
                              comp.vbar)),
        g4_calibration([r for r in n1 if r["phenotype"] == MARGIN], n2[0]),
        g5_sharpness(lam_m, n5, MARGIN),
    ]
    rows = list(n1) + list(n2) + list(n5) + list(n6)
    rows += [{"null": "N4", "phenotype": r["phenotype"], "terms": r["terms"],
              "mean": r["lambda_after"]["mean"], "sd": r["lambda_after"]["sd"],
              "n": r["lambda_after"]["n"],
              "lambda_before": r["lambda_before"]["mean"]} for r in n4]
    return {"nulls": rows, "n4_detail": n4, "null_sd": null_sd,
            "gates": gate_table(gates)}
