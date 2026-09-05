r"""The six nulls, each answering a specific way the headline could be an artifact.

    N1  synthetic populations at a known true rbar_E, giving the calibration
        curve that turns a raw Lambda into a corrected one
    N2  seed-label permutation, drawn independently per trait and applied to both
        halves of that trait
    N3  re-randomisation of the A/B item split
    N4  the attenuation the pooled-slope mediation induces when there is nothing
        for it to mediate, which is the reference a partialled Lambda is read against
    N5  a run-level multiplicative gain at rbar_E = 0, the sharpness artifact
    N6  a run-batch offset at rbar_E = 0, the batch artifact

N2 deserves its warning.  Permuting the seed labels jointly across traits, one
permutation for all of them, leaves every cross-trait deviation product intact
and returns a Lambda centred on the observed value; it looks like a null and is
not one.  The permutation must be drawn independently per trait, and the same
permutation must then be applied to both halves of that trait, or the cross-half
product is destroyed along with the correlation and the null becomes trivially
centred on zero instead.
"""

from __future__ import annotations

import numpy as np

from seednoise.estimator import estimate, sigma_e, nearest_psd
from seednoise.mediation import fit_mediation, residualise
from seednoise.population import ACCURACY, MARGIN, Population
from seednoise.reliability import variance_components
from seednoise.simulate import SimSpec, default_spec, simulate

__all__ = ["n1_calibration", "permute_seed_labels", "n2_permutation",
           "n3_resplit", "n4_zero_mediation", "n5_gain_artifact",
           "n6_batch_offset", "summarise", "NOMINAL_ITEMS"]

DEFAULT_RBARS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50)

# Item count used by null N4 when the population does not carry its own.
NOMINAL_ITEMS = 2000


def summarise(values, truth=None) -> dict:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan")}
    out = {"n": int(v.size), "mean": float(v.mean()),
           "sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
           "q025": float(np.percentile(v, 2.5)),
           "q50": float(np.percentile(v, 50)),
           "q975": float(np.percentile(v, 97.5)),
           "q95": float(np.percentile(v, 95))}
    if truth is not None:
        out["truth"] = float(truth)
        out["bias"] = float(v.mean() - truth)
    return out


def n1_calibration(rbars=DEFAULT_RBARS, n_rep=2000, n_config=85, seed=0,
                   phenotypes=(MARGIN, ACCURACY), which=("all",), progress=None,
                   **spec_kw) -> list:
    """Calibration curve: what the estimator returns at each known truth.

    The curve is the object the paper uses twice, once to subtract a calibration
    offset from the reported ``R_E`` and once to supply the null SD that every
    power statement rests on.
    """
    rows = []
    for r in rbars:
        got = {(p, w): [] for p in phenotypes for w in which}
        for i in range(n_rep):
            spec = default_spec(rbar_e=r, n_config=n_config,
                                seed=seed + 100003 * i, **spec_kw)
            pop = simulate(spec)
            for p, w in got:
                got[(p, w)].append(
                    estimate(pop, p, which=w, check=False).lambda_hat)
            if progress is not None and (i + 1) % progress == 0:
                print(f"  N1 rbar={r}: {i + 1}/{n_rep}", flush=True)
        K = len(default_spec(rbar_e=r, **spec_kw).traits)
        truth = float(np.sqrt(1.0 + (K - 1) * r))
        for (p, w), v in got.items():
            # The batch-free contrast has one direction rather than R - 1, so its
            # own calibration curve is the only fair reference for it.
            rows.append({"null": "N1", "rbar_E_true": r, "phenotype": p,
                         "contrast": w, **summarise(v, truth)})
    return rows


def permute_seed_labels(pop: Population, name: str, rng) -> Population:
    """Independently permute the run labels within each trait, on both halves.

    The same permutation goes to half A and half B of a trait so the cross-half
    pairing survives; a different permutation per half would break the identity
    rather than the correlation, which is a different and useless null.
    """
    ph = pop.pheno(name)
    A, B = ph.A.copy(), ph.B.copy()
    for j in range(pop.K):
        for c in range(pop.N):
            pi = rng.permutation(pop.R)
            A[c, :, j] = ph.A[c, pi, j]
            B[c, :, j] = ph.B[c, pi, j]
    return pop.with_phenotype(f"{name}|perm", A, B)


def n2_permutation(pop: Population, name: str, n_rep=2000, seed=0) -> dict:
    """Lambda under per-trait seed-label permutation, whose truth is 1."""
    rng = np.random.default_rng(seed)
    observed = estimate(pop, name, check=False).lambda_hat
    out = []
    for _ in range(n_rep):
        p = permute_seed_labels(pop, name, rng)
        out.append(estimate(p, f"{name}|perm", check=False).lambda_hat)
    # The permuted mean is the reference the observed value is read against, so
    # both travel together and a row can never be read without its baseline.
    frac = float(np.mean([v >= observed for v in out])) if out else float("nan")
    return {"null": "N2", "phenotype": name, "observed": float(observed),
            "p_perm": frac, **summarise(out, 1.0)}


def n3_resplit(rebuild, name: str, n_rep=200, seed=0) -> dict:
    """Lambda across re-randomisations of the A/B item split.

    ``rebuild(split_seed)`` returns the population scored under a fresh split, so
    this null needs item-level data and is skipped, loudly, without it.
    """
    out = []
    for i in range(n_rep):
        pop = rebuild(seed + i)
        out.append(estimate(pop, name, check=False).lambda_hat)
    return {"null": "N3", "phenotype": name, **summarise(out)}


def n4_zero_mediation(pop: Population, name: str, n_rep=500, seed=0,
                      terms=("x", "s")) -> dict:
    """The attenuation the pooled-slope regression induces on matched data.

    Populations are simulated with the observed ``Sigma_E`` and an independent
    gain covariate, so nothing in them is genuinely mediated by sharpness; what
    the regression removes here is the reference against which the real
    post-partialling ``Lambda`` is judged.
    """
    S = sigma_e(pop, name)
    sd = np.sqrt(np.clip(np.diag(S), 1e-9, None))
    Rm = S / np.outer(sd, sd)
    np.fill_diagonal(Rm, 1.0)
    Rm = np.where(np.isfinite(Rm), Rm, 0.0)
    # Sampling needs a covariance, and the unbiased estimate need not be one.
    target, clipped = nearest_psd(np.outer(sd, sd) * Rm, floor=1e-12)
    # The reference populations must carry the observed item noise as well as the
    # observed seed covariance, or the attenuation measured here belongs to a more
    # reliable population than the one being corrected.  A population that does not
    # know its item counts gets a nominal count with the item SD solved to match.
    comp = variance_components(pop, name)
    n_items = (tuple([NOMINAL_ITEMS] * pop.K) if pop.n_items is None
               else tuple(int(n) for n in np.asarray(pop.n_items).tolist()))
    n_half = np.maximum(np.asarray(n_items, dtype=np.float64) / 2.0, 1.0)
    item_sd = tuple(np.sqrt(np.clip(comp.noise2, 1e-12, None) * n_half).tolist())

    before, after = [], []
    for i in range(n_rep):
        spec = SimSpec(n_config=pop.N, n_runs=pop.R, traits=tuple(pop.traits),
                       n_items=n_items, item_sd=item_sd, seed=seed + 7919 * i)
        spec.sigma_e = target
        sim = simulate(spec)
        before.append(estimate(sim, name, check=False).lambda_hat)
        fit = fit_mediation(sim, name, terms=terms)
        res = residualise(sim, name, fit)
        after.append(estimate(res, list(res.phenotypes)[-1], check=False).lambda_hat)
    b, a = np.asarray(before), np.asarray(after)
    keep = np.isfinite(b) & np.isfinite(a) & (b > 0)
    return {"null": "N4", "phenotype": name, "terms": "+".join(terms),
            "psd_clipped_mass": clipped,
            "lambda_before": summarise(b), "lambda_after": summarise(a),
            "retained_fraction": summarise(
                (a[keep] - 1.0) / np.where(b[keep] > 1.0, b[keep] - 1.0, np.nan))}


def n5_gain_artifact(gain_sd=0.05, n_rep=2000, n_config=85, seed=0,
                     **spec_kw) -> list:
    """How much Lambda a run-level multiplicative gain manufactures at rbar_E = 0.

    The signature the paper looks for is asymmetry: sharpness inflates the margin
    arm hard and the accuracy arm barely, since scaling margins away from zero
    pushes items across the boundary in both directions.
    """
    got = {MARGIN: [], ACCURACY: []}
    for i in range(n_rep):
        spec = default_spec(rbar_e=0.0, n_config=n_config,
                            seed=seed + 104729 * i, **spec_kw)
        spec.gain_sd = gain_sd
        pop = simulate(spec)
        for p in got:
            got[p].append(estimate(pop, p, check=False).lambda_hat)
    return [{"null": "N5", "gain_sd": gain_sd, "phenotype": p,
             **summarise(v, 1.0)} for p, v in got.items()]


def n6_batch_offset(offset_sd=0.05, n_rep=2000, n_config=85, seed=0,
                    **spec_kw) -> list:
    """What a run-batch offset manufactures, and that the batch-free contrast is immune."""
    rows = []
    got = {(p, w): [] for p in (MARGIN, ACCURACY) for w in ("all", "batch_free")}
    for i in range(n_rep):
        spec = default_spec(rbar_e=0.0, n_config=n_config,
                            seed=seed + 1299709 * i, **spec_kw)
        spec.batch_offset_sd = offset_sd
        pop = simulate(spec)
        for (p, w) in got:
            got[(p, w)].append(
                estimate(pop, p, which=w, check=False).lambda_hat)
    for (p, w), v in got.items():
        rows.append({"null": "N6", "offset_sd": offset_sd, "phenotype": p,
                     "contrast": w, **summarise(v, 1.0)})
    return rows
