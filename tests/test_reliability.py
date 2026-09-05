"""Variance components, the information ratio, and the power arithmetic."""

from __future__ import annotations

import numpy as np
import pytest

from seednoise.population import MARGIN
from seednoise.reliability import (
    aggregate_reliability, information_ratio, power_at, power_table,
    variance_components,
)
from tests.conftest import gaussian_population


def test_the_components_recover_a_planted_split():
    """Seed variance one, item noise a quarter, on a population built that way."""
    pop, Sig = gaussian_population(n_config=4000, K=4, rbar=0.0, sd=np.ones(4),
                                   item_noise=0.5, seed=0)
    c = variance_components(pop, MARGIN)
    assert np.abs(c.sigma_e2 - 1.0).max() < 0.1
    assert np.abs(c.noise2 - 0.25).max() < 0.05
    assert np.abs(c.reliability - 0.8).max() < 0.03


def test_vbar_is_the_ratio_of_the_two_totals():
    pop, _ = gaussian_population(n_config=2000, K=4, item_noise=0.5, seed=1)
    c = variance_components(pop, MARGIN)
    assert c.vbar == pytest.approx(c.noise2.sum() / c.sigma_e2.sum())


def test_a_negative_seed_variance_is_reported_not_clipped():
    """Clipping would claim a reliability of one where the estimate is noise."""
    pop, _ = gaussian_population(n_config=30, K=4, sd=np.full(4, 1e-6),
                                 item_noise=2.0, seed=2)
    c = variance_components(pop, MARGIN)
    assert (c.sigma_e2 < 0).any()
    assert np.isfinite(c.noise2).all()


def test_the_information_ratio_matches_the_papers_worked_value():
    assert float(information_ratio(0.35)) == pytest.approx(1.658, abs=0.002)


def test_the_information_ratio_is_smallest_at_a_half():
    p = np.array([0.5, 0.35, 0.25])
    r = information_ratio(p)
    assert r[0] == pytest.approx(np.pi / 2, abs=1e-6)
    assert r[0] < r[1] < r[2]


def test_a_trait_at_floor_or_ceiling_has_no_ratio():
    with pytest.raises(ValueError, match="strictly inside"):
        information_ratio([0.5, 1.0])


def test_a_measured_boundary_overrides_the_gaussian_one():
    assert information_ratio(0.35, z0=0.0) == pytest.approx(0.35 * 0.65 * 2 * np.pi)


def test_aggregate_reliability_rises_with_lambda():
    assert aggregate_reliability(1.0, 0.5) == pytest.approx(1 / 1.5)
    assert aggregate_reliability(2.0, 0.5) > aggregate_reliability(1.0, 0.5)


def test_power_is_alpha_at_the_null_and_rises_with_the_alternative():
    assert power_at(1.0, 0.05) == pytest.approx(0.05, abs=1e-6)
    assert power_at(1.2, 0.05) > power_at(1.1, 0.05) > 0.05


def test_the_power_table_converts_rbar_to_lambda_before_reporting():
    rows = power_table(0.0518, K=10, rbars=(0.115,))
    assert rows[0]["Lambda"] == pytest.approx(np.sqrt(1 + 9 * 0.115))
    assert 0.5 < rows[0]["power"] <= 1.0


def test_power_uses_the_alternative_sd_when_it_is_given():
    tight = power_at(1.35, 0.05, alt_sd=0.05)
    loose = power_at(1.35, 0.05, alt_sd=0.20)
    assert tight > loose
