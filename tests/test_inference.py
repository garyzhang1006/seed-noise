"""Intervals on the ratio, with the clustering the design actually has.

The estimand is a ratio of sums, so every interval here is built on the same
linearised influence and the square root is applied last, which is exact because
the map is monotone.
"""

from __future__ import annotations

import numpy as np
import pytest

from seednoise.inference import (
    cluster_influence, cluster_se, cluster_t_interval, config_bootstrap, deff,
    icc, wild_bootstrap_t,
)


def _data(n_clusters=17, per=5, theta=2.25, noise=0.15, seed=0, cluster_sd=0.0):
    rng = np.random.default_rng(seed)
    N = n_clusters * per
    cluster = np.repeat(np.arange(n_clusters), per)
    U = rng.uniform(0.8, 1.2, N)
    shift = rng.standard_normal(n_clusters)[cluster] * cluster_sd
    T = (theta + shift) * U + rng.standard_normal(N) * noise
    return T, U, cluster


def test_the_influences_sum_to_zero_by_construction():
    T, U, g = _data()
    psi, keys, _ = cluster_influence(T, U, g)
    assert psi.sum() == pytest.approx(0.0, abs=1e-12)
    assert keys.size == 17


def test_the_standard_error_falls_as_the_root_of_the_cluster_count():
    se_small = cluster_se(*_data(n_clusters=8, seed=1))
    se_large = cluster_se(*_data(n_clusters=128, seed=1))
    assert se_large < se_small
    assert 2.0 < se_small / se_large < 8.0


def test_the_point_estimate_is_the_square_root_of_the_ratio():
    T, U, g = _data(theta=2.25)
    i = cluster_t_interval(T, U, g)
    assert i.point == pytest.approx(np.sqrt(T.sum() / U.sum()))
    assert i.lo < i.point < i.hi


def test_the_untransformed_interval_is_on_theta_itself():
    T, U, g = _data()
    i = cluster_t_interval(T, U, g, sqrt_transform=False)
    assert i.point == pytest.approx(T.sum() / U.sum())


def test_the_square_root_preserves_the_interval_because_it_is_monotone():
    T, U, g = _data()
    raw = cluster_t_interval(T, U, g, sqrt_transform=False)
    rt = cluster_t_interval(T, U, g)
    assert rt.lo == pytest.approx(np.sqrt(raw.lo))
    assert rt.hi == pytest.approx(np.sqrt(raw.hi))


def test_a_single_cluster_is_refused_by_both_clustered_intervals():
    T, U, _ = _data()
    one = np.zeros(T.size, dtype=int)
    for fn in (cluster_t_interval, wild_bootstrap_t):
        with pytest.raises(ValueError, match="at least two"):
            fn(T, U, one)


def test_the_wild_bootstrap_is_reproducible_and_close_to_the_t_interval():
    T, U, g = _data(seed=2)
    a = wild_bootstrap_t(T, U, g, n_boot=999, seed=3)
    b = wild_bootstrap_t(T, U, g, n_boot=999, seed=3)
    assert (a.lo, a.hi) == (b.lo, b.hi)
    t = cluster_t_interval(T, U, g)
    assert abs(a.lo - t.lo) < 0.15 * (t.hi - t.lo)
    assert a.n_clusters == 17


def test_mismatched_inputs_are_refused():
    T, U, g = _data()
    with pytest.raises(ValueError, match="matching 1-D arrays"):
        wild_bootstrap_t(T, U[:-1], g[:-1])


def test_the_wild_bootstrap_covers_at_about_its_nominal_rate():
    """The point of the bootstrap-t is coverage at seventeen clusters, so measure it."""
    theta, hits, n_rep = 2.25, 0, 200
    for i in range(n_rep):
        T, U, g = _data(seed=100 + i, cluster_sd=0.25)
        iv = wild_bootstrap_t(T, U, g, n_boot=399, seed=i)
        hits += int(iv.lo <= np.sqrt(theta) <= iv.hi)
    assert 0.88 <= hits / n_rep <= 0.995


def test_ignoring_the_clustering_gives_a_narrower_interval_when_clusters_matter():
    T, U, g = _data(cluster_sd=0.4, seed=5)
    clustered = cluster_t_interval(T, U, g)
    naive = config_bootstrap(T, U, n_boot=999, seed=0)
    assert (naive.hi - naive.lo) < (clustered.hi - clustered.lo)


def test_icc_is_near_zero_without_cluster_structure_and_high_with_it():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(20), 5)
    flat = rng.standard_normal(100)
    lumpy = rng.standard_normal(20)[g] * 3.0 + rng.standard_normal(100) * 0.2
    assert abs(icc(flat, g)) < 0.2
    assert icc(lumpy, g) > 0.8


def test_icc_refuses_a_design_it_cannot_estimate_from():
    with pytest.raises(ValueError, match="at least two clusters"):
        icc(np.arange(5.0), np.zeros(5, dtype=int))


def test_the_design_effect_is_one_plus_four_rho_at_five_runs_per_recipe():
    assert deff(0.0) == pytest.approx(1.0)
    assert deff(0.15) == pytest.approx(1.6)
    assert deff(0.15, m=3.0) == pytest.approx(1.3)
