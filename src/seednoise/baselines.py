r"""The pre-committed bake-off, with four ways for the full matrix to lose.

Every model predicts the same held-out quantity, the directly observed aggregate
seed SD on recipes it was not fitted on, and every model is handed the held-out
per-trait seed SDs.  Only the correlation structure differs between them, which
is what makes the comparison a test of whether cross-benchmark correlation must
be modelled rather than a test of whose variance estimates are better.

    P0   independence, the correlation is the identity, current practice
    P1   rank-one competence fitted to the training correlation
    P1g  rank-two, competence plus a gain axis
    P2   the Cheverud plug-in, substituting the free phenotypic matrix R_P
    P3   the full training estimate of R_E
"""

from __future__ import annotations


import numpy as np

from seednoise.estimator import (
    _T_from, contrast_basis, correlation, phenotypic_correlation, sigma_e,
)
from seednoise.population import Population

__all__ = ["BASELINES", "predict_sigma_agg", "observed_sigma_agg", "bakeoff"]


def _low_rank(R: np.ndarray, rank: int) -> np.ndarray:
    """Best rank-``k`` correlation approximation, renormalised to unit diagonal."""
    R = np.asarray(R, dtype=np.float64)
    w, V = np.linalg.eigh(0.5 * (R + R.T))
    order = np.argsort(w)[::-1]
    w, V = np.clip(w[order][:rank], 0.0, None), V[:, order][:, :rank]
    A = V * np.sqrt(w)
    C = A @ A.T
    resid = np.clip(1.0 - np.diag(C), 0.0, None)
    C = C + np.diag(resid)
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    return C / np.outer(d, d)


def _corr_P0(train, test, name):
    return np.eye(train.K)


def _corr_P1(train, test, name):
    return _low_rank(correlation(sigma_e(train, name)), 1)


def _corr_P1g(train, test, name):
    return _low_rank(correlation(sigma_e(train, name)), 2)


def _corr_P2(train, test, name):
    # R_P needs no replicates at all, which is the point of the Cheverud question.
    return phenotypic_correlation(test, name)


def _corr_P3(train, test, name):
    return correlation(sigma_e(train, name))


BASELINES = {
    "P0": ("independence", _corr_P0, 0),
    "P1": ("rank-one competence", _corr_P1, None),
    "P1g": ("rank-one plus gain", _corr_P1g, None),
    "P2": ("Cheverud plug-in with R_P", _corr_P2, 0),
    "P3": ("full matrix", _corr_P3, None),
}


def observed_sigma_agg(pop: Population, name: str) -> float:
    """``sqrt(mean_c T_c)``: the aggregate seed SD, measured with no model at all."""
    ph = pop.pheno(name)
    T = _T_from(ph.A, ph.B, contrast_basis(pop.R), pop.K)
    m = float(T.mean())
    return float(np.sqrt(m)) if m > 0 else float("nan")


def predict_sigma_agg(train: Population, test: Population, name: str,
                      model: str) -> float:
    """One model's prediction of the held-out aggregate seed SD."""
    if model not in BASELINES:
        raise ValueError(f"unknown baseline {model!r}; have {sorted(BASELINES)}")
    _, corr_fn, _ = BASELINES[model]
    sd = np.sqrt(np.clip(np.diag(sigma_e(test, name)), 0.0, None))
    R = np.asarray(corr_fn(train, test, name), dtype=np.float64)
    if R.shape != (test.K, test.K):
        raise ValueError(f"{model}: correlation is {R.shape}, expected square K")
    R = np.where(np.isfinite(R), R, 0.0)
    np.fill_diagonal(R, 1.0)
    S = np.outer(sd, sd) * R
    v = float(S.sum()) / test.K ** 2
    return float(np.sqrt(v)) if v > 0 else float("nan")


def bakeoff(pop: Population, name: str, n_folds: int = 5, seed: int = 0) -> list:
    """Cluster-held-out bake-off scored by squared error of ``log sigma_agg``.

    Folds are drawn over recipe clusters rather than configurations, so no recipe
    is ever in both the fitting and the scoring set; splitting on configurations
    would let a model see sibling runs of the recipe it is being scored on.
    """
    recipes = np.unique(pop.recipe)
    if recipes.size < n_folds:
        raise ValueError(
            f"{recipes.size} recipe clusters cannot be split into {n_folds} folds"
        )
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(recipes), n_folds)
    rows = []
    for model, (label, _, npar) in BASELINES.items():
        errs, preds = [], []
        for held in folds:
            test = pop.subset_clusters(held)
            train = pop.subset_clusters(np.setdiff1d(recipes, held))
            obs = observed_sigma_agg(test, name)
            pred = predict_sigma_agg(train, test, name, model)
            if not (np.isfinite(obs) and np.isfinite(pred) and obs > 0 and pred > 0):
                continue
            errs.append((np.log(pred) - np.log(obs)) ** 2)
            preds.append(pred)
        rows.append({
            "model": model, "label": label, "n_free": npar,
            "mse_log_sigma_agg": float(np.mean(errs)) if errs else float("nan"),
            "n_folds_scored": len(errs),
            "mean_predicted_sigma_agg": float(np.mean(preds)) if preds else float("nan"),
        })
    return rows
