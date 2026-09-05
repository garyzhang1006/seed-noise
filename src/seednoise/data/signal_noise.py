r"""The zero-cost external check, on Heineman et al.'s signal-and-noise release.

That release includes ``random_seeds``: eleven OLMo2 1B runs at 5xC that differ
only in the training seed, nine more that differ only in the data order, and a
task-by-step table of their scores.  Nothing has to be trained or scored to use
it, which makes it the cheapest independent test of the paper's central claim
that replicate seeds move the whole battery together rather than each task on its
own.

Two things separate it from the main estimate and neither can be repaired from
the released table, so both are stated rather than worked around.  The scores are
aggregates, so there is no item-level half split and no way to subtract the item
noise; the correlations are therefore attenuated, and ``Lambda`` computed here is
a floor on the seed correlation, not an estimate of it.  And a run-level
sharpness factor is not separable from a genuine seed factor at the aggregate
level, which is the same confound the main analysis handles with mediation, so
the leave-one-task-out competence proxy is offered here as a partial control.

The estimand is the same one throughout: with ``rbar`` the mean off-diagonal
correlation of the task scores across replicate runs, ``Lambda^2 = 1 + (K-1) rbar``
and ``K_eff = K / Lambda^2``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["HF_REPO", "PARQUET", "parquet_url", "download_random_seeds",
           "load_random_seeds", "score_matrix", "rbar_from_matrix",
           "external_check"]

HF_REPO = "allenai/signal-and-noise"
PARQUET = "data/random_seeds-00000-of-00001.parquet"

# 'seed' varies the training seed; 'data' varies the data order.  'high-eval' is
# one run evaluated on a denser step grid and is not a replicate set.
RUN_TYPES = ("seed", "data")


def parquet_url(file: str = PARQUET) -> str:
    return f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{file}"


def download_random_seeds(dest, file: str = PARQUET, timeout: int = 120):
    """Fetch the parquet once; about 3.4 MB, and the only network this arm needs."""
    import urllib.request
    from pathlib import Path

    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = parquet_url(file)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, open(dest, "wb") as f:
            f.write(r.read())
    except Exception as e:                                    # noqa: BLE001
        raise RuntimeError(f"could not download {url}: {e}") from e
    return dest


def load_random_seeds(path=None, dest="cache/random_seeds.parquet"):
    """The released table, with the columns this module needs checked once."""
    import pandas as pd

    p = path if path is not None else download_random_seeds(dest)
    df = pd.read_parquet(p)
    need = {"run_name", "step", "value", "run_type", "task_name", "metric"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(
            f"{p} is missing {sorted(missing)}; the random_seeds table carries "
            f"{sorted(need)}"
        )
    return df


def score_matrix(df, run_type: str = "seed", metric: str = "bits_per_byte",
                 n_steps: int = 20, tasks=None):
    """An ``(R, K, S)`` array of replicate runs by task by step.

    Only steps every replicate reached are used, because a run scored at a step
    its siblings never reached is a run at a different token budget, and the last
    ``n_steps`` of those are kept so the estimate sits near the end of training
    rather than averaging over the part of the curve where the runs are still
    separating.
    """
    d = df[(df["run_type"] == run_type) & (df["metric"] == metric)]
    if tasks is not None:
        d = d[d["task_name"].isin(list(tasks))]
    if d.empty:
        raise ValueError(
            f"no rows with run_type={run_type!r} and metric={metric!r}; the table "
            f"has run types {sorted(df['run_type'].unique())} and metrics "
            f"{sorted(df['metric'].unique())}"
        )
    runs = sorted(d["run_name"].unique())
    if len(runs) < 3:
        raise ValueError(f"{len(runs)} replicate run(s) for {run_type!r}: too few")
    shared = None
    for r in runs:
        s = set(d.loc[d["run_name"] == r, "step"])
        shared = s if shared is None else (shared & s)
    if not shared:
        raise ValueError(f"the {len(runs)} {run_type!r} runs share no evaluation step")
    steps = sorted(shared)[-int(n_steps):]

    wide = (d[d["step"].isin(steps)]
            .pivot_table(index=["run_name", "step"], columns="task_name",
                         values="value"))
    wide = wide.dropna(axis=1, how="any")
    task_names = list(wide.columns)
    if len(task_names) < 2:
        raise ValueError(
            f"only {len(task_names)} task(s) are scored on every run and step, so "
            "there is no cross-task correlation to take"
        )
    X = np.stack([wide.xs(s, level="step").reindex(runs).to_numpy()
                  for s in steps], axis=-1)
    return X, runs, task_names, steps


def _loo_mean(Z):
    """Leave-one-task-out mean, the competence proxy x^(-j), per run."""
    K = Z.shape[1]
    tot = Z.sum(axis=1, keepdims=True)
    return (tot - Z) / (K - 1)


def rbar_from_matrix(X, partial_out: str = "none"):
    """Mean off-diagonal correlation and its ``Lambda``, averaged over steps.

    Averaging correlations over steps rather than pooling runs across steps keeps
    each estimate within one token budget, which is what stops a shared training
    curve from entering the correlation as if it were a seed effect.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"expected (runs, tasks, steps), got {X.shape}")
    R, K, S = X.shape
    if partial_out not in ("none", "loo_mean"):
        raise ValueError(f"partial_out must be 'none' or 'loo_mean', got {partial_out!r}")
    rbars, dropped = [], 0
    for s in range(S):
        Z = X[:, :, s] - X[:, :, s].mean(axis=0)
        if partial_out == "loo_mean":
            P = _loo_mean(Z)
            # One slope per task, fitted across runs: the aggregate-level twin of
            # the paper's mediation on the competence proxy.
            denom = (P ** 2).sum(axis=0)
            b = np.divide((P * Z).sum(axis=0), denom, out=np.zeros(K),
                          where=denom > 0)
            Z = Z - b * P
        sd = Z.std(axis=0, ddof=1)
        keep = sd > 0
        if keep.sum() < 2:
            dropped += 1
            continue
        C = np.corrcoef(Z[:, keep].T)
        off = C[~np.eye(int(keep.sum()), dtype=bool)]
        rbars.append(float(np.nanmean(off)))
    if not rbars:
        raise ValueError("every step was degenerate; no correlation could be taken")
    rbar = float(np.mean(rbars))
    lam = float(np.sqrt(max(1.0 + (K - 1) * rbar, 0.0)))
    return {"rbar": rbar, "rbar_sd_over_steps": float(np.std(rbars, ddof=1))
            if len(rbars) > 1 else float("nan"),
            "Lambda": lam, "K_eff": K / lam ** 2 if lam > 0 else float("inf"),
            "R": R, "K": K, "n_steps": len(rbars), "degenerate_steps": dropped,
            "partial_out": partial_out}


def external_check(df, n_steps: int = 20, metrics=("bits_per_byte", "acc_per_char"),
                   run_types=RUN_TYPES):
    """One row per replicate set, metric and control, ready for the table.

    Every number here is attenuated by item noise the release does not let us
    subtract, so a row above one is evidence of a seed factor and a row at one is
    evidence of nothing either way.
    """
    rows = []
    for rt in run_types:
        for metric in metrics:
            try:
                X, runs, tasks, steps = score_matrix(df, rt, metric, n_steps)
            except ValueError as e:
                rows.append({"run_type": rt, "metric": metric, "error": str(e)})
                continue
            for control in ("none", "loo_mean"):
                r = rbar_from_matrix(X, partial_out=control)
                rows.append({"run_type": rt, "metric": metric, **r,
                             "first_step": int(steps[0]), "last_step": int(steps[-1]),
                             "tasks": " ".join(tasks)})
    return rows
