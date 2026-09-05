"""The external check, on a table whose answer is known, then on the real one.

The real release is 3.4 MB and behind the network, so it is exercised only when
``SEEDNOISE_SN_PARQUET`` points at a downloaded copy; everything else here runs
on a synthetic table with a planted seed correlation.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from seednoise.data.signal_noise import (
    external_check, load_random_seeds, parquet_url, rbar_from_matrix, score_matrix,
)

TASKS = [f"t{j}" for j in range(8)]


def _table(rbar=0.2, R=10, S=25, noise=1.0, seed=0, run_type="seed",
           metric="bits_per_byte", missing_task=None):
    """Scores with an equicorrelated run factor and independent task noise."""
    rng = np.random.default_rng(seed)
    K = len(TASKS)
    rows = []
    for s in range(S):
        common = rng.standard_normal(R)
        own = rng.standard_normal((R, K))
        val = (np.sqrt(rbar) * common[:, None] + np.sqrt(1 - rbar) * own
               + noise * rng.standard_normal((R, K)) * 0.0)
        for i in range(R):
            for j, t in enumerate(TASKS):
                if missing_task is not None and t == missing_task and i == 0:
                    continue
                rows.append({"run_name": f"run{i}", "step": 1000 * (s + 1),
                             "value": float(val[i, j]), "run_type": run_type,
                             "task_name": t, "metric": metric})
    return pd.DataFrame(rows)


def test_the_url_points_at_the_released_parquet():
    assert parquet_url().endswith("data/random_seeds-00000-of-00001.parquet")
    assert "allenai/signal-and-noise" in parquet_url()


def test_a_planted_correlation_is_recovered():
    X, runs, tasks, steps = score_matrix(_table(rbar=0.3, S=40), n_steps=40)
    assert X.shape == (10, 8, 40)
    r = rbar_from_matrix(X)
    assert abs(r["rbar"] - 0.3) < 0.05
    assert r["Lambda"] == pytest.approx(np.sqrt(1 + 7 * r["rbar"]))
    assert r["K_eff"] == pytest.approx(8 / r["Lambda"] ** 2)


def test_independent_tasks_land_at_one_rather_than_below_it():
    r = rbar_from_matrix(score_matrix(_table(rbar=0.0, S=40), n_steps=40)[0])
    assert abs(r["rbar"]) < 0.05
    assert abs(r["Lambda"] - 1.0) < 0.15


def test_only_the_last_steps_are_used_and_they_are_shared(tmp_path):
    df = _table(S=30)
    df = df[~((df["run_name"] == "run3") & (df["step"] == 30000))]
    X, runs, tasks, steps = score_matrix(df, n_steps=5)
    assert steps == [25000, 26000, 27000, 28000, 29000]     # 30000 is not shared
    assert X.shape[2] == 5


def test_a_task_missing_from_one_run_is_dropped_rather_than_imputed():
    X, runs, tasks, steps = score_matrix(_table(missing_task="t3"), n_steps=5)
    assert "t3" not in tasks and len(tasks) == 7


def test_the_competence_control_removes_a_planted_common_factor():
    """A pure run-level shift is exactly what the leave-one-out proxy absorbs."""
    X = score_matrix(_table(rbar=0.6, S=40), n_steps=40)[0]
    raw = rbar_from_matrix(X)["rbar"]
    controlled = rbar_from_matrix(X, partial_out="loo_mean")["rbar"]
    assert raw > 0.5 and controlled < raw


def test_an_unknown_control_is_refused():
    X = score_matrix(_table(S=5), n_steps=5)[0]
    with pytest.raises(ValueError, match="partial_out"):
        rbar_from_matrix(X, partial_out="regression")


def test_a_missing_metric_names_what_the_table_does_have():
    with pytest.raises(ValueError, match="metrics"):
        score_matrix(_table(), metric="perplexity")


def test_too_few_replicate_runs_is_refused():
    with pytest.raises(ValueError, match="too few"):
        score_matrix(_table(R=2), n_steps=5)


def test_a_table_without_the_needed_columns_says_which(tmp_path):
    p = tmp_path / "bad.parquet"
    pd.DataFrame({"run_name": ["a"], "value": [1.0]}).to_parquet(p)
    with pytest.raises(KeyError, match="metric"):
        load_random_seeds(p)


def test_the_check_returns_one_row_per_set_metric_and_control():
    df = pd.concat([_table(rbar=0.25, S=25, run_type="seed"),
                    _table(rbar=0.1, S=25, run_type="data", seed=1)])
    rows = external_check(df, n_steps=10, metrics=("bits_per_byte",))
    assert len(rows) == 4
    assert {(r["run_type"], r["partial_out"]) for r in rows} == {
        ("seed", "none"), ("seed", "loo_mean"),
        ("data", "none"), ("data", "loo_mean")}
    seed_raw = next(r for r in rows
                    if r["run_type"] == "seed" and r["partial_out"] == "none")
    assert seed_raw["Lambda"] > 1.3
    assert seed_raw["first_step"] < seed_raw["last_step"]


def test_a_metric_that_is_absent_becomes_a_row_with_its_reason():
    rows = external_check(_table(S=5), n_steps=5, metrics=("acc_per_char",),
                          run_types=("seed",))
    assert len(rows) == 1 and "no rows" in rows[0]["error"]


@pytest.mark.skipif(not os.environ.get("SEEDNOISE_SN_PARQUET"),
                    reason="set SEEDNOISE_SN_PARQUET to the downloaded parquet")
def test_the_real_release_shows_a_seed_factor_in_the_continuous_phenotype():
    df = load_random_seeds(os.environ["SEEDNOISE_SN_PARQUET"])
    rows = external_check(df, n_steps=20)
    raw = next(r for r in rows if r["run_type"] == "seed"
               and r["metric"] == "bits_per_byte" and r["partial_out"] == "none")
    assert raw["R"] == 10 and raw["K"] == 18
    assert raw["Lambda"] > 1.3                    # attenuated, and still above one
    ctl = next(r for r in rows if r["run_type"] == "seed"
               and r["metric"] == "bits_per_byte" and r["partial_out"] == "loo_mean")
    assert ctl["Lambda"] > 1.2                    # it survives the competence control
