r"""The nine gates, each with the fallback that leaves a complete paper standing.

A gate is not an assertion.  It is a decision recorded before the data was
opened, together with what the paper does when the condition fails, so that a
failure changes the claim rather than the analysis.  Every gate therefore
reports what it measured even when it passes, and every fallback is a sentence
the paper can print.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from seednoise.estimator import half_asymmetry, sigma_e
from seednoise.population import MARGIN, Population
from seednoise.reliability import information_ratio, power_at

__all__ = ["Gate", "g0_coverage", "g1_accuracy_power", "g2_information_ratio",
           "g3_cross_half_symmetry", "g4_calibration", "g5_sharpness",
           "g6_transport", "g7_heldout_loss", "g8_batch_free", "gate_table"]


@dataclass
class Gate:
    name: str
    condition: str
    fallback: str
    measured: dict = field(default_factory=dict)
    passed: bool = False

    def as_row(self) -> dict:
        return {"gate": self.name, "condition": self.condition,
                "passed": self.passed, "fallback": self.fallback,
                **{f"m_{k}": v for k, v in self.measured.items()}}


def g0_coverage(n_cells_parsed: int, n_cells_expected: int = 125,
                n_estimation: int = 85, min_estimation: int = 70) -> Gate:
    """Per-item outputs stream and parse for every cell at the common step."""
    g = Gate("G0", "per-item outputs parse for all 125 cells at the common step",
             "drop the cell; below 70 estimation configurations, report at reduced N")
    g.measured = {"cells_parsed": int(n_cells_parsed),
                  "cells_expected": int(n_cells_expected),
                  "estimation_configs": int(n_estimation),
                  "min_estimation": int(min_estimation)}
    g.passed = (n_cells_parsed == n_cells_expected
                and n_estimation >= min_estimation)
    return g


def g1_accuracy_power(null_sd: float, rho_g: float, rbar_target: float = 0.115,
                      K: int = 10, min_power: float = 0.80) -> Gate:
    """Accuracy power at the field-relevant threshold, from the measured reliability."""
    g = Gate("G1", f"accuracy power at rbar_E={rbar_target} is at least {min_power}",
             "accuracy demoted to secondary; the abstract states the margin only")
    lam = float(np.sqrt(1.0 + (K - 1) * rbar_target))
    pw = power_at(lam, null_sd)
    g.measured = {"lambda_alt": lam, "null_sd": float(null_sd),
                  "rho_g": float(rho_g), "power": pw}
    g.passed = bool(np.isfinite(pw) and pw >= min_power)
    return g


def g2_information_ratio(accuracy_by_trait, z0=None, minimum: float = 1.4) -> Gate:
    """The margin's information advantage, on the battery median."""
    g = Gate("G2", f"information ratio at least {minimum} on the battery median",
             "the margin loses its more-informative framing; both arms reported flat")
    ratios = information_ratio(np.asarray(accuracy_by_trait, dtype=float), z0)
    med = float(np.median(ratios))
    g.measured = {"median_ratio": med, "min_ratio": float(np.min(ratios)),
                  "max_ratio": float(np.max(ratios))}
    g.passed = med >= minimum
    return g


def g3_cross_half_symmetry(pop: Population, name: str = MARGIN,
                           max_z: float = 4.0) -> Gate:
    """Cross-half covariance symmetric, and every diagonal entry positive.

    Asymmetry is judged against its own sampling noise rather than against a fixed
    fraction of the matrix scale, because the cross-half product at ``N`` in the
    low hundreds is noisy enough that any fixed fraction either fires constantly
    or never fires, depending only on the phenotype's units and the sample size.
    The threshold is a z score on the largest off-diagonal entry, and ``max_z`` is
    set high enough to absorb the ``K(K-1)/2`` comparisons being made at once.
    """
    g = Gate("G3", "cross-half covariance is symmetric with a positive diagonal",
             "re-split by source document; if still asymmetric, report leakage")
    S = sigma_e(pop, name)
    Dn = half_asymmetry(pop, name, per_config=True)
    D = Dn.mean(axis=0)
    se = Dn.std(axis=0, ddof=1) / np.sqrt(Dn.shape[0])
    off = ~np.eye(pop.K, dtype=bool)
    z = np.abs(D[off]) / np.where(se[off] > 0, se[off], np.inf)
    scale = float(np.sqrt(np.mean(np.clip(np.diag(S), 0, None) ** 2))) or 1.0
    n_neg = int((np.diag(S) <= 0).sum())
    g.measured = {"max_z_asymmetry": float(z.max()) if z.size else 0.0,
                  "max_relative_asymmetry": float(np.abs(D).max() / scale),
                  "negative_diagonals": n_neg,
                  "min_diagonal": float(np.min(np.diag(S)))}
    g.passed = bool((z.size == 0 or z.max() <= max_z) and n_neg == 0)
    return g


def g4_calibration(n1_rows, n2_row, tol: float = 0.02) -> Gate:
    """The estimator returns 1 at the null, under both the simulation and the permutation."""
    g = Gate("G4", f"|Lambda - 1| at most {tol} under N1 at zero and under N2",
             "subtract the calibrated offset; report both raw and corrected")
    at_zero = [r for r in n1_rows
               if r.get("rbar_E_true") == 0.0 and r.get("contrast", "all") == "all"]
    if not at_zero:
        raise ValueError("G4 needs an N1 row at rbar_E = 0 on the full contrast set")
    off_sim = max(abs(r["mean"] - 1.0) for r in at_zero)
    off_perm = abs(float(n2_row["mean"]) - 1.0)
    g.measured = {"offset_n1": off_sim, "offset_n2": off_perm,
                  "null_sd": float(np.mean([r["sd"] for r in at_zero]))}
    g.passed = off_sim <= tol and off_perm <= tol
    return g


def g5_sharpness(observed_lambda: float, n5_rows, phenotype: str = MARGIN,
                 max_share: float = 0.50) -> Gate:
    """The sharpness null must not manufacture half the observed excess."""
    g = Gate("G5", f"N5 manufactures less than {max_share:.0%} of the observed excess",
             "sharpness becomes the headline; Lambda_A is reported alone")
    row = next((r for r in n5_rows if r["phenotype"] == phenotype), None)
    if row is None:
        raise ValueError(f"G5 has no N5 row for phenotype {phenotype!r}")
    excess = float(observed_lambda) - 1.0
    manufactured = float(row["mean"]) - 1.0
    share = manufactured / excess if excess > 0 else float("inf")
    g.measured = {"observed_excess": excess, "manufactured_excess": manufactured,
                  "share": share}
    g.passed = bool(np.isfinite(share) and share < max_share)
    return g


def g6_transport(lambda_main: float, lambda_arm2: float,
                 tol: float = 0.20) -> Gate:
    """PolyPythias reproduces the DataDecide estimate within a fifth."""
    g = Gate("G6", f"PolyPythias transport within {tol:.0%}",
             "transport reported as a failure, with the ratio")
    ratio = (float(lambda_arm2) - 1.0) / (float(lambda_main) - 1.0) \
        if lambda_main > 1 else float("nan")
    g.measured = {"lambda_main": float(lambda_main),
                  "lambda_arm2": float(lambda_arm2), "excess_ratio": ratio}
    g.passed = bool(np.isfinite(ratio) and abs(ratio - 1.0) <= tol)
    return g


def g7_heldout_loss(has_field: bool, field_name: str = "") -> Gate:
    """A held-out-loss field would give a competence mediator that is not a proxy."""
    g = Gate("G7", "a held-out-loss field exists in the DataDecide metadata",
             "the competence mediator is the leave-one-trait-out proxy x^(-j)")
    g.measured = {"has_field": bool(has_field), "field": field_name}
    g.passed = bool(has_field)
    return g


def g8_batch_free(lambda_full: float, lambda_free: float,
                  share: float = 0.5) -> Gate:
    """The batch-free contrast must retain at least half the excess."""
    g = Gate("G8", "batch-free Lambda is at least 1 + 0.5 (Lambda_full - 1)",
             "the batch-free Lambda becomes the headline; the full set moves to the appendix")
    need = 1.0 + share * (float(lambda_full) - 1.0)
    g.measured = {"lambda_full": float(lambda_full),
                  "lambda_batch_free": float(lambda_free), "threshold": need}
    g.passed = bool(np.isfinite(lambda_free) and lambda_free >= need)
    return g


def gate_table(gates) -> list:
    return [g.as_row() for g in gates]
