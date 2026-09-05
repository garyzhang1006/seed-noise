"""Each gate on both sides: the condition it is meant to pass, and the failure
it was registered to catch.

A gate that cannot fail is decoration, so every test here has a partner that
drives the same gate to False, and every fallback string is checked to be the
sentence the paper would print in that case.
"""

from __future__ import annotations

import numpy as np
import pytest

from seednoise.gates import (
    g0_coverage, g1_accuracy_power, g2_information_ratio, g3_cross_half_symmetry,
    g4_calibration, g5_sharpness, g6_transport, g7_heldout_loss, g8_batch_free,
    gate_table,
)
from seednoise.population import ACCURACY, MARGIN, Phenotype, Population
from seednoise.simulate import default_spec, simulate


@pytest.fixture(scope="module")
def clean_pop():
    return simulate(default_spec(rbar_e=0.15, n_config=85, seed=11))


def test_g0_passes_on_a_complete_parse():
    g = g0_coverage(125, 125, 85)
    assert g.passed and g.measured["cells_parsed"] == 125


def test_g0_fails_when_estimation_configurations_fall_below_the_floor():
    assert not g0_coverage(125, 125, 60).passed
    assert not g0_coverage(118, 125, 85).passed


def test_g1_passes_when_the_null_is_tight_enough_to_see_the_alternative():
    g = g1_accuracy_power(null_sd=0.04, rho_g=0.3)
    assert g.passed and g.measured["power"] > 0.80
    assert g.measured["lambda_alt"] == pytest.approx(np.sqrt(1 + 9 * 0.115))


def test_g1_fails_on_a_wide_null_and_demotes_accuracy():
    g = g1_accuracy_power(null_sd=0.60, rho_g=0.3)
    assert not g.passed and g.measured["power"] < 0.80
    assert "demoted to secondary" in g.fallback


def test_g2_reproduces_the_registered_information_ratio():
    """At p = 0.35 the paper claims 1.66; the gate must agree with it."""
    g = g2_information_ratio([0.35] * 10)
    assert g.measured["median_ratio"] == pytest.approx(1.66, abs=0.01)
    assert g.passed


def test_g2_fails_when_the_measured_boundary_sits_at_the_density_peak():
    """Under the Gaussian reading the ratio never drops below pi/2, so only a
    measured z_0 that disagrees with the accuracy can fail this gate."""
    g = g2_information_ratio([0.2] * 10, z0=0.0)
    assert not g.passed
    assert g.measured["median_ratio"] < 1.4


def test_g3_passes_on_a_population_whose_halves_are_exchangeable(clean_pop):
    g = g3_cross_half_symmetry(clean_pop)
    assert g.passed
    assert g.measured["negative_diagonals"] == 0
    assert g.measured["max_z_asymmetry"] < 4.0


def test_g3_catches_a_half_that_carries_its_own_signal(clean_pop):
    """Leakage makes the cross-half product asymmetric far beyond its own noise."""
    ph = clean_pop.pheno(MARGIN)
    B = np.array(ph.B, dtype=np.float64)
    B[:, :, 1:] += 3.0 * ph.A[:, :, :-1]        # trait j leaks into trait j+1 in B only
    leaky = Population({MARGIN: Phenotype(MARGIN, ph.A, B)},
                       traits=list(clean_pop.traits))
    g = g3_cross_half_symmetry(leaky)
    assert not g.passed
    assert g.measured["max_z_asymmetry"] > 4.0
    assert "re-split by source document" in g.fallback


def _n1(rbar, mean, sd=0.05, contrast="all"):
    return {"rbar_E_true": rbar, "contrast": contrast, "mean": mean, "sd": sd}


def test_g4_passes_when_both_nulls_sit_at_one():
    g = g4_calibration([_n1(0.0, 1.004), _n1(0.2, 1.67)], {"mean": 0.997})
    assert g.passed and g.measured["offset_n1"] == pytest.approx(0.004)


def test_g4_fails_when_either_null_is_off_centre():
    assert not g4_calibration([_n1(0.0, 1.05)], {"mean": 1.0}).passed
    assert not g4_calibration([_n1(0.0, 1.0)], {"mean": 1.05}).passed


def test_g4_refuses_to_judge_without_a_row_at_the_null():
    with pytest.raises(ValueError, match="rbar_E = 0"):
        g4_calibration([_n1(0.2, 1.67)], {"mean": 1.0})


def test_g5_passes_when_sharpness_explains_a_minority_of_the_excess():
    g = g5_sharpness(1.60, [{"phenotype": MARGIN, "mean": 1.15}])
    assert g.passed and g.measured["share"] == pytest.approx(0.25)


def test_g5_fails_when_sharpness_manufactures_most_of_it():
    g = g5_sharpness(1.20, [{"phenotype": MARGIN, "mean": 1.15}])
    assert not g.passed
    assert "sharpness becomes the headline" in g.fallback


def test_g5_needs_the_row_for_the_phenotype_it_was_asked_about():
    with pytest.raises(ValueError, match="no N5 row"):
        g5_sharpness(1.6, [{"phenotype": ACCURACY, "mean": 1.1}], phenotype=MARGIN)


def test_g6_compares_excess_rather_than_level():
    """1.5 against 1.55 is a 10% gap in excess, not the 3% gap in level."""
    g = g6_transport(1.50, 1.55)
    assert g.measured["excess_ratio"] == pytest.approx(1.10)
    assert g.passed
    assert not g6_transport(1.50, 1.90).passed


def test_g6_is_undefined_when_the_main_estimate_shows_no_excess():
    g = g6_transport(1.0, 1.4)
    assert not g.passed and not np.isfinite(g.measured["excess_ratio"])


def test_g7_names_the_proxy_it_falls_back_to():
    assert g7_heldout_loss(True, "eval/loss").passed
    g = g7_heldout_loss(False)
    assert not g.passed and "leave-one-trait-out proxy" in g.fallback


def test_g8_passes_when_the_batch_free_contrast_keeps_half_the_excess():
    g = g8_batch_free(1.60, 1.35)
    assert g.passed and g.measured["threshold"] == pytest.approx(1.30)
    assert not g8_batch_free(1.60, 1.25).passed


def test_g8_fails_on_a_non_finite_batch_free_estimate():
    assert not g8_batch_free(1.6, float("nan")).passed


def test_the_gate_table_is_flat_and_carries_every_measurement():
    rows = gate_table([g0_coverage(125, 125, 85), g7_heldout_loss(False)])
    assert [r["gate"] for r in rows] == ["G0", "G7"]
    assert rows[0]["m_cells_parsed"] == 125
    assert all(isinstance(v, (int, float, str, bool)) for r in rows for v in r.values())
