r"""Cluster-aware intervals for a ratio of sums, which is what ``Lambda`` is.

The unit of analysis is the configuration, but configurations sharing a data
recipe are not independent, so every interval is clustered on the recipe.  With
17 estimation recipes the asymptotics are thin, which is why the pre-registered
interval is a wild cluster bootstrap-t with ``t(G-1)`` critical values (``t(16)``
as registered, ``t(24)`` on the 25 recipes the release was actually run on)
rather than a normal approximation, and why a plain configuration bootstrap is
reported beside it rather than instead of it.

Intervals are formed on ``theta = sum_c T_c / sum_c U_c`` and then square-rooted.
Because the square root is monotone, transforming the endpoints is exact, which a
delta-method interval on ``Lambda`` itself would not be.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t

__all__ = ["Interval", "cluster_influence", "cluster_se", "wild_bootstrap_t",
           "config_bootstrap", "icc", "deff"]


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    se: float
    method: str
    n_clusters: int

    def contains(self, x: float) -> bool:
        return bool(self.lo <= x <= self.hi)

    def as_row(self) -> dict:
        return {"point": self.point, "lo": self.lo, "hi": self.hi,
                "se": self.se, "method": self.method,
                "n_clusters": self.n_clusters}


def _theta(T, U):
    sU = float(np.sum(U))
    if sU == 0:
        return float("nan")
    return float(np.sum(T)) / sU


def cluster_influence(T, U, cluster):
    """Per-cluster influence of the ratio, ``sum_{c in g}(T_c - theta U_c)/sum U``.

    This is the linearisation the cluster-robust SE and the wild bootstrap both
    use, so the two agree on what a cluster contributes.
    """
    T, U = np.asarray(T, float), np.asarray(U, float)
    g = np.asarray(cluster)
    th = _theta(T, U)
    r = T - th * U
    keys, inv = np.unique(g, return_inverse=True)
    e = np.bincount(inv, weights=r, minlength=keys.size)
    return e / float(np.sum(U)), keys, inv


def cluster_se(T, U, cluster) -> float:
    psi, _, _ = cluster_influence(T, U, cluster)
    return float(np.sqrt(np.sum(psi ** 2)))


def wild_bootstrap_t(T, U, cluster, n_boot=4999, alpha=0.05, seed=0,
                     sqrt_transform=True) -> Interval:
    """Wild cluster bootstrap-t on the ratio, with Rademacher cluster weights.

    The bootstrap imposes the point estimate and perturbs each cluster's residual
    by a sign, then recomputes both the statistic and its standard error on the
    bootstrap sample, which is what makes it a bootstrap-t rather than a
    percentile method and what gives it its small-cluster accuracy.
    """
    T, U = np.asarray(T, float), np.asarray(U, float)
    g = np.asarray(cluster)
    if T.shape != U.shape or T.ndim != 1:
        raise ValueError(f"T {T.shape} and U {U.shape} must be matching 1-D arrays")
    keys, inv = np.unique(g, return_inverse=True)
    G = keys.size
    if G < 2:
        raise ValueError(f"{G} cluster(s): a clustered interval needs at least two")
    if n_boot < 199:
        raise ValueError(
            f"n_boot={n_boot}: a two-sided 95% bootstrap-t needs at least 199 "
            "draws for its tail quantiles to exist at all; pass n_boot=4999 for "
            "the registered setting, or use cluster_t_interval for a fast check"
        )
    th = _theta(T, U)
    sU = float(np.sum(U))
    r = T - th * U
    se = cluster_se(T, U, g)

    rng = np.random.default_rng(seed)
    v = rng.choice([-1.0, 1.0], size=(n_boot, G))
    vb = v[:, inv]                                   # (B, N) weight per config
    Tb = th * U + vb * r
    th_b = Tb.sum(axis=1) / sU
    rb = Tb - th_b[:, None] * U
    # Cluster sums of the bootstrap residuals, one row per bootstrap draw.
    eb = np.zeros((n_boot, G))
    np.add.at(eb, (np.arange(n_boot)[:, None], np.broadcast_to(inv, vb.shape)), rb)
    se_b = np.sqrt((eb ** 2).sum(axis=1)) / sU
    ok = se_b > 0
    tb = np.full(n_boot, np.nan)
    tb[ok] = (th_b[ok] - th) / se_b[ok]
    tb = tb[np.isfinite(tb)]
    if tb.size < max(100, int(0.9 * n_boot)):
        raise RuntimeError(
            f"only {tb.size} of {n_boot} bootstrap draws produced a finite t; the "
            "cluster structure is too degenerate for a bootstrap-t interval"
        )
    lo_q, hi_q = np.percentile(tb, [100 * (1 - alpha / 2), 100 * (alpha / 2)])
    lo, hi = th - lo_q * se, th - hi_q * se
    if sqrt_transform:
        th, lo, hi = (float(np.sqrt(x)) if x >= 0 else float("nan")
                      for x in (th, lo, hi))
        se = se / (2 * th) if np.isfinite(th) and th > 0 else float("nan")
    return Interval(th, lo, hi, float(se), f"wild cluster bootstrap-t (G={G})", G)


def cluster_t_interval(T, U, cluster, alpha=0.05, sqrt_transform=True) -> Interval:
    """Cluster-robust SE with ``t(G-1)`` critical values, the pre-registered form."""
    T, U = np.asarray(T, float), np.asarray(U, float)
    keys = np.unique(np.asarray(cluster))
    G = keys.size
    if G < 2:
        raise ValueError(f"{G} cluster(s): a clustered interval needs at least two")
    th = _theta(T, U)
    se = cluster_se(T, U, cluster)
    crit = float(student_t.ppf(1 - alpha / 2, G - 1))
    lo, hi = th - crit * se, th + crit * se
    if sqrt_transform:
        th, lo, hi = (float(np.sqrt(x)) if x >= 0 else float("nan")
                      for x in (th, lo, hi))
        se = se / (2 * th) if np.isfinite(th) and th > 0 else float("nan")
    return Interval(th, lo, hi, float(se), f"cluster-robust t({G - 1})", G)


def config_bootstrap(T, U, n_boot=4999, alpha=0.05, seed=0,
                     sqrt_transform=True) -> Interval:
    """Percentile bootstrap resampling configurations, ignoring the clustering.

    Reported beside the clustered interval, never instead of it: it is the
    interval a reader who believes configurations are independent would compute,
    and the gap between the two is the design effect made visible.
    """
    T, U = np.asarray(T, float), np.asarray(U, float)
    N = T.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, size=(n_boot, N))
    th_b = T[idx].sum(axis=1) / U[idx].sum(axis=1)
    th = _theta(T, U)
    lo, hi = np.percentile(th_b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    se = float(np.std(th_b, ddof=1))
    if sqrt_transform:
        th, lo, hi = (float(np.sqrt(x)) if x >= 0 else float("nan")
                      for x in (th, lo, hi))
        se = se / (2 * th) if np.isfinite(th) and th > 0 else float("nan")
    return Interval(th, float(lo), float(hi), se,
                    "configuration percentile bootstrap", N)


def icc(values, cluster) -> float:
    """One-way random-effects ICC of a per-configuration statistic across recipes."""
    x = np.asarray(values, dtype=np.float64)
    keys, inv = np.unique(np.asarray(cluster), return_inverse=True)
    G = keys.size
    n = np.bincount(inv)
    if G < 2 or x.size <= G:
        raise ValueError(
            f"ICC needs at least two clusters and more units than clusters, got "
            f"{G} clusters over {x.size} units"
        )
    gm = np.bincount(inv, weights=x) / n
    msb = float(np.sum(n * (gm - x.mean()) ** 2) / (G - 1))
    msw = float(np.sum((x - gm[inv]) ** 2) / (x.size - G))
    n0 = (x.size - np.sum(n ** 2) / x.size) / (G - 1)
    denom = msb + (n0 - 1) * msw
    return float((msb - msw) / denom) if denom > 0 else float("nan")


def deff(rho_icc: float, m: float = 5.0) -> float:
    """``1 + (m - 1) rho``: the variance inflation from clustering at size ``m``."""
    return float(1.0 + (m - 1.0) * float(rho_icc))
