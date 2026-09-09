"""The command line and the four experiments, exercised end to end offline.

These are the paths a user actually runs, so what is checked here is that they
exit cleanly, write the files the paper cites by name, and refuse bad arguments
with a message that says what to do instead.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from seednoise.artifacts import write_json, write_table
from seednoise.cli import build_parser, main
from seednoise.experiments import run_bakeoff, run_nulls, run_primary, run_screen
from seednoise.population import ACCURACY, MARGIN
from seednoise.simulate import default_spec, simulate


@pytest.fixture(scope="module")
def small_pop():
    spec = default_spec(rbar_e=0.2, n_config=40, seed=3)
    spec.n_items = tuple(200 for _ in spec.n_items)
    return simulate(spec)


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------
# artifacts


def test_a_table_carries_the_union_of_every_row_key(tmp_path):
    p = write_table(tmp_path, "demo", [{"a": 1}, {"b": 2.5}])
    rows = _rows(p)
    assert p.name == "tab_demo.csv"
    assert set(rows[0]) == {"a", "b"}
    assert rows[1]["b"] == "2.5" and rows[1]["a"] == ""
    assert _rows(write_table(tmp_path, "flags", [{"ok": True, "n": np.int64(3)}]))[0] \
        == {"ok": "yes", "n": "3"}


def test_numpy_values_survive_the_json_writer(tmp_path):
    p = write_json(tmp_path, "m", {"x": np.arange(3), "y": np.float32(1.5),
                                   "z": {"k": np.int64(7)}})
    got = json.loads(p.read_text())
    assert got == {"x": [0, 1, 2], "y": 1.5, "z": {"k": 7}}


def test_an_empty_table_is_still_written_so_the_run_is_auditable(tmp_path):
    p = write_table(tmp_path, "empty", [])
    assert p.exists()


# --------------------------------------------------------------------------
# experiments


def test_e1_screens_the_battery_and_reports_its_gates(small_pop):
    e1 = run_screen(small_pop, n_cells_parsed=125, n_estimation=small_pop.N)
    assert [g["gate"] for g in e1["gates"]] == ["G0", "G2", "G3", "G7"]
    assert len(e1["reliability"]) == 2 * small_pop.K      # one row per phenotype
    lam = {r["phenotype"]: r["Lambda"] for r in e1["screen_lambda"]}
    assert lam[MARGIN] > 1.0 and lam[ACCURACY] > 1.0
    assert np.asarray(e1["R_E_margin"]).shape == (small_pop.K, small_pop.K)
    # G7 cannot pass on a simulated population: there is no held-out loss field.
    assert not next(g for g in e1["gates"] if g["gate"] == "G7")["passed"]


def test_e2_returns_both_phenotypes_with_three_intervals_each(small_pop):
    e2 = run_primary(small_pop, n_boot=299, seed=1)
    head = [r for r in e2["primary"] if r["label"] == "full contrast set"]
    assert {r["phenotype"] for r in head} == {MARGIN, ACCURACY}
    for r in head:
        assert r["Lambda"] > 1.0
        assert r["wild_lo"] < r["Lambda"] < r["wild_hi"]
        assert r["K_eff"] == pytest.approx(small_pop.K / r["Lambda"] ** 2)
    assert {r["label"] for r in e2["primary"]} >= {"batch-free contrast"}
    # One pooled slope per trait, for each of the three mediator sets and both
    # phenotypes: 3 x 2 x K rows.
    assert len(e2["mediation"]) == 6 * small_pop.K
    assert {r["terms"] for r in e2["mediation"]} == {"x", "s", "x+s"}
    assert set(e2["matrices"]) == {MARGIN, ACCURACY}
    assert set(e2["matrices"][MARGIN]) >= {"Sigma_E", "R_E", "K_eff", "DEFF"}


def test_the_practitioner_table_is_a_monotone_flip_probability(small_pop):
    e2 = run_primary(small_pop, n_boot=299, seed=1)
    p = [r["P_flip"] for r in e2["practitioner"]]
    assert all(0.0 <= x <= 0.5 for x in p)
    assert all(a >= b for a, b in zip(p, p[1:]))      # larger gap, rarer flip


def test_e3_runs_every_null_and_lands_them_where_registered(small_pop):
    e3 = run_nulls(small_pop, n_rep=60, n_rep_slow=20, rbars=(0.0, 0.2), seed=2)
    names = {r["null"] for r in e3["nulls"]}
    assert names == {"N1", "N2", "N4", "N5", "N6"}
    zero = [r for r in e3["nulls"]
            if r["null"] == "N1" and r.get("rbar_E_true") == 0.0
            and r.get("contrast") == "all"]
    assert zero and abs(zero[0]["mean"] - 1.0) < 0.05
    assert [g["gate"] for g in e3["gates"]] == ["G1", "G4", "G5"]


def test_e4_ranks_a_structured_model_above_the_diagonal_one(small_pop):
    e4 = run_bakeoff(small_pop, n_folds=3, seed=4)
    assert e4["winner_margin"] in {"P1", "P1g", "P2", "P3"}
    assert {r["model"] for r in e4["bakeoff"]} >= {"P0", "P1", "P3"}
    assert "frobenius_RE_minus_RP" in e4["cheverud"][0]


# --------------------------------------------------------------------------
# command line


def test_the_parser_exposes_every_documented_subcommand():
    for cmd in ("download", "reduce", "fetch", "analyze", "selftest"):
        assert build_parser().parse_args([cmd]).cmd == cmd


def test_no_subcommand_prints_usage_rather_than_a_traceback(capsys):
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 2
    assert "usage" in capsys.readouterr().err.lower()


def test_analyze_on_a_synthetic_population_writes_every_table(tmp_path):
    out = tmp_path / "results"
    code = main(["analyze", "--synthetic", "--fast", "--skip-nulls",
                 "--rbar", "0.2", "--n-config", "40", "--n-boot", "299",
                 "--out", str(out), "--quiet"])
    assert code == 0
    for t in ("primary", "practitioner", "mediation", "reliability",
              "bakeoff", "cheverud", "gates"):
        assert (out / f"tab_{t}.csv").exists(), t
    assert json.loads((out / "source.json").read_text())["source"] == "synthetic"
    lam = [float(r["Lambda"]) for r in _rows(out / "tab_primary.csv")
           if r["label"] == "full contrast set" and r["phenotype"] == MARGIN]
    assert lam and lam[0] > 1.2                       # rbar_E = 0.2 was planted


def test_analyze_reports_the_gate_that_a_synthetic_population_cannot_pass(tmp_path):
    out = tmp_path / "r"
    main(["analyze", "--synthetic", "--fast", "--skip-nulls", "--n-config", "40",
          "--n-boot", "299", "--out", str(out), "--quiet"])
    gates = {r["gate"]: r["passed"] for r in _rows(out / "tab_gates.csv")}
    assert gates["G7"] == "no" and gates["G3"] == "yes"


def test_analyze_without_runs_says_which_command_makes_them(tmp_path):
    with pytest.raises(FileNotFoundError, match="seednoise"):
        main(["analyze", "--runs", str(tmp_path / "nothing"), "--quiet"])


def test_selftest_passes_offline_and_writes_its_evidence(tmp_path):
    out = tmp_path / "st"
    assert main(["selftest", "--n-rep", "40", "--out", str(out)]) == 0
    ev = json.loads((out / "selftest.json").read_text())
    assert ev["failures"] == []
    assert abs(ev["lambda"] - ev["target"]) < 0.12
    for t in ("gates", "primary", "reliability", "bakeoff"):
        assert (out / f"tab_{t}.csv").exists(), t


def test_a_bootstrap_too_small_to_have_tails_says_so_instead_of_guessing():
    """99 draws cannot carry a 2.5% quantile; the old guard called that degenerate."""
    from seednoise.inference import wild_bootstrap_t
    rng = np.random.default_rng(0)
    T, U = rng.random(40) + 1.0, rng.random(40) + 1.0
    cl = np.repeat(np.arange(8), 5)
    with pytest.raises(ValueError, match="at least 199"):
        wild_bootstrap_t(T, U, cl, n_boot=99)
    assert np.isfinite(wild_bootstrap_t(T, U, cl, n_boot=999).lo)


# --------------------------------------------------------------------------
# the two side arms


def _seed_table(rbar=0.25, R=10, S=12, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    K, rows = 6, []
    for s in range(S):
        common = rng.standard_normal(R)
        own = rng.standard_normal((R, K))
        v = np.sqrt(rbar) * common[:, None] + np.sqrt(1 - rbar) * own
        for i in range(R):
            for j in range(K):
                rows.append({"run_name": f"r{i}", "step": 1000 * (s + 1),
                             "value": float(v[i, j]), "run_type": "seed",
                             "task_name": f"t{j}", "metric": "bits_per_byte"})
    return pd.DataFrame(rows)


def test_external_writes_its_table_from_a_local_parquet(tmp_path):
    pq = tmp_path / "rs.parquet"
    _seed_table().to_parquet(pq)
    out = tmp_path / "res"
    assert main(["external", "--parquet", str(pq), "--out", str(out),
                 "--n-steps", "12"]) == 0
    rows = _rows(out / "tab_external.csv")
    raw = [r for r in rows if r["partial_out"] == "none"]
    assert raw and float(raw[0]["Lambda"]) > 1.3


def test_analyze_adds_the_transport_gate_when_arm_two_runs_are_present(tmp_path):
    from seednoise.phenotypes import ItemPhenotypes
    from seednoise.store import save_run

    runs = tmp_path / "arm2"
    rng = np.random.default_rng(0)
    trait = np.repeat(np.arange(10), 30)
    for si, size in enumerate(("70m", "160m", "410m")):
        base = rng.standard_normal(trait.size)
        for k in range(3):
            # A run-level shift shared by every trait is what makes Lambda > 1,
            # so the arm-2 population has a transportable seed factor to find.
            m = (base + 0.2 * si + 0.6 * rng.standard_normal()
                 + 0.5 * rng.standard_normal(trait.size))
            items = ItemPhenotypes(np.arange(trait.size), trait, m, m > 0)
            save_run(runs / f"{size}-{k}.npz", items,
                     {"recipe": f"pythia-{size}", "size": size, "seed": k,
                      "step": 143000, "batch": k, "gain": -1.0 - 0.01 * k})
    out = tmp_path / "res"
    assert main(["analyze", "--synthetic", "--fast", "--skip-nulls",
                 "--n-config", "40", "--n-boot", "299", "--arm2-runs", str(runs),
                 "--arm2-n-runs", "3", "--out", str(out), "--quiet"]) == 0
    gates = {r["gate"]: r for r in _rows(out / "tab_gates.csv")}
    assert "G6" in gates
    assert gates["G6"]["m_lambda_arm2"] != ""
    assert json.loads((out / "arm2_source.json").read_text())["n_config"] == 3


def test_the_arm_two_parser_defaults_to_the_released_seeds():
    a = build_parser().parse_args(["arm2"])
    assert a.sizes == ["70m", "160m", "410m"]
    assert a.seeds == [str(s) for s in range(1, 10)]
    assert a.revision == "step143000" and a.max_tokens == 30_000


def test_arm2_manifest_names_are_distinct_per_subset():
    from seednoise.cli import _arm2_manifest_name
    from seednoise.data.polypythias import SEEDS, SIZES
    assert _arm2_manifest_name(SIZES, [str(s) for s in SEEDS]) == "arm2_manifest"
    a = _arm2_manifest_name(["70m"], ["1"])
    b = _arm2_manifest_name(["70m"], ["2"])
    assert a != b and a == "arm2_manifest__70m__seed-1"
