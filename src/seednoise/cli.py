"""The command line the paper's pipeline is driven from.

Every stage is a separate subcommand because the expensive one is the download
and it must be restartable, and because the analysis has to be runnable on a
synthetic population before anyone has spent a hundred gigabytes of bandwidth.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from seednoise import __version__
from seednoise.artifacts import write_json, write_table
from seednoise.population import ACCURACY, MARGIN

__all__ = ["main", "build_parser"]

DEFAULT_RECIPES = None      # None means every recipe in the release

# Imported by name so `seednoise arm2 --help` works without torch installed.
from seednoise.data.polypythias import FINAL_REVISION as PP_REVISION  # noqa: E402
from seednoise.data.polypythias import SEEDS as PP_SEEDS  # noqa: E402
from seednoise.data.polypythias import SIZES as PP_SIZES  # noqa: E402
from seednoise.experiments.e5_sensitivity import GAIN_GRID  # noqa: E402


def _log(msg):
    print(f"[seednoise] {msg}", flush=True)


# --------------------------------------------------------------------------
# population sources


def _synthetic(args):
    from seednoise.simulate import default_spec, simulate
    spec = default_spec(rbar_e=args.rbar, n_config=args.n_config, seed=args.seed,
                        gain_sd=args.gain_sd, batch_offset_sd=args.offset_sd)
    if args.fast:
        spec.n_items = tuple(max(200, n // 20) for n in spec.n_items)
    return simulate(spec)


def _population(args):
    if getattr(args, "synthetic", False):
        _log(f"synthetic population, rbar_E={args.rbar}, N={args.n_config}")
        return _synthetic(args), {"source": "synthetic", "rbar_E_true": args.rbar}
    from seednoise.build import build_population
    from seednoise.data.datadecide import TRAITS
    pop, info = build_population(args.runs, TRAITS, n_runs=args.n_runs)
    _log(f"built {pop.N} configurations from {info['n_runs_read']} runs "
         f"in {args.runs}")
    if info["dropped_cells"]:
        _log(f"dropped {len(info['dropped_cells'])} incomplete cells, first "
             f"{info['dropped_cells'][:3]}")
    return pop, {"source": str(args.runs), **info}


# --------------------------------------------------------------------------
# subcommands


def cmd_download(args):
    from seednoise.data.datadecide import RECIPES, download_recipe
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for r in (args.recipes or RECIPES):
        download_recipe(r, dest / f"{r}.tar", progress=not args.quiet)
    return 0


def cmd_reduce(args):
    from seednoise.data.datadecide import reduce_recipe
    out = Path(args.out)
    rows = []
    for tar in sorted(Path(args.tars).glob("*.tar")):
        res = reduce_recipe(tar, out, sizes=args.sizes, progress=not args.quiet)
        _log(f"{res['recipe']}: {len(res['runs'])} runs, {len(res['skipped'])} skipped")
        rows.append(res)
    write_json(out, "reduce_manifest", rows)
    return 0


def cmd_fetch(args):
    """Download, reduce and delete one recipe at a time, so peak disk stays small."""
    from seednoise.data.datadecide import RECIPES, download_recipe, reduce_recipe
    tmp, out = Path(args.tmp), Path(args.out)
    tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in (args.recipes or RECIPES):
        tar = tmp / f"{r}.tar"
        t0 = time.time()
        download_recipe(r, tar, progress=not args.quiet)
        res = reduce_recipe(tar, out, sizes=args.sizes, progress=not args.quiet)
        if not args.keep_tars:
            tar.unlink(missing_ok=True)
        _log(f"{r}: {len(res['runs'])} runs in {time.time() - t0:.0f}s")
        rows.append(res)
        write_json(out, "reduce_manifest", rows)
    return 0


def cmd_external(args):
    """The zero-cost external check on the signal-and-noise random-seed runs."""
    from seednoise.data.signal_noise import external_check, load_random_seeds

    out = Path(args.out)
    df = load_random_seeds(args.parquet, dest=args.cache)
    rows = external_check(df, n_steps=args.n_steps)
    write_table(out, "external", rows)
    for r in rows:
        if "error" in r:
            _log(f"{r['run_type']}/{r['metric']}: {r['error']}")
            continue
        _log(f"{r['run_type']}/{r['metric']} control={r['partial_out']}: "
             f"rbar={r['rbar']:+.4f} Lambda={r['Lambda']:.4f} "
             f"K_eff={r['K_eff']:.2f} over {r['n_steps']} steps")
    _log("these correlations are attenuated by item noise the release does not "
         "let us subtract, so each Lambda is a floor")
    return 0


def cmd_arm2(args):
    """Score the PolyPythias replicate seeds; the only stage that needs a GPU."""
    from seednoise.data.polypythias import run_arm2

    res = run_arm2(args.out, sizes=args.sizes, seeds=[int(s) for s in args.seeds],
                   n_per_task=args.n_per_task, revision=args.revision,
                   max_tokens=args.max_tokens, tasks=args.tasks,
                   device=args.device, progress=None if args.quiet else 20)
    write_json(args.out, "arm2_manifest", res["runs"])
    _log(f"{len(res['runs'])} runs over {res['n_items']} items written to {args.out}")
    return 0


def cmd_analyze(args):
    pop, source = _population(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _log(f"N={pop.N} R={pop.R} K={pop.K} clusters={pop.n_clusters}")

    from seednoise.experiments import run_bakeoff, run_nulls, run_primary, run_screen

    e1 = run_screen(pop, n_cells_parsed=source.get("n_cells", 125),
                    n_estimation=pop.N)
    write_table(out, "reliability", e1["reliability"])
    _log("E1 gates: " + ", ".join(f"{g['gate']}={'pass' if g['passed'] else 'FAIL'}"
                                  for g in e1["gates"]))

    e2 = run_primary(pop, n_boot=args.n_boot, seed=args.seed)
    write_table(out, "primary", e2["primary"])
    write_table(out, "practitioner", e2["practitioner"])
    write_table(out, "mediation", e2["mediation"])
    for name, m in e2["matrices"].items():
        write_json(out, f"matrices_{name}", m)
    head = [r for r in e2["primary"] if r["label"] == "full contrast set"]
    for r in head:
        if "wild_lo" in r:
            ci = f"[{r['wild_lo']:.4f}, {r['wild_hi']:.4f}]"
        else:
            ci = f"(no wild interval: {r.get('wild_error', 'unknown reason')})"
        _log(f"Lambda[{r['phenotype']}] = {r['Lambda']:.4f} {ci}  "
             f"K_eff = {r['K_eff']:.2f}")

    gates = list(e1["gates"])
    if not args.skip_nulls:
        e3 = run_nulls(pop, n_rep=args.n_rep, n_rep_slow=max(50, args.n_rep // 4),
                       seed=args.seed, progress=None if args.quiet else 20)
        write_table(out, "nulls", e3["nulls"])
        gates += e3["gates"]

    e4 = run_bakeoff(pop, seed=args.seed)
    write_table(out, "bakeoff", e4["bakeoff"])
    write_table(out, "cheverud", e4["cheverud"])
    _log(f"bake-off winner (margin): {e4['winner_margin']}")

    if args.arm2_runs:
        from seednoise.build import build_population
        from seednoise.data.datadecide import TRAITS
        from seednoise.estimator import estimate
        from seednoise.gates import g6_transport
        arm2, a_info = build_population(args.arm2_runs, TRAITS,
                                        n_runs=args.arm2_n_runs)
        lam2 = estimate(arm2, MARGIN, check=True).lambda_hat
        head_margin = next(r["Lambda"] for r in head if r["phenotype"] == MARGIN)
        g6 = g6_transport(head_margin, lam2)
        gates.append(g6.as_row())
        _log(f"arm 2: {arm2.N} configurations, Lambda={lam2:.4f}, "
             f"G6={'pass' if g6.passed else 'FAIL'}")
        write_json(out, "arm2_source", a_info)

    write_table(out, "gates", gates)
    write_json(out, "source", source)
    failed = [g["gate"] for g in gates if not g["passed"]]
    _log("gates failed: " + (", ".join(failed) if failed else "none"))
    _log(f"tables written to {out}")
    return 0


def cmd_sensitivity(args):
    """E5: the gain calibration, the re-splits and the leave-one-out sweep."""
    from seednoise.build import build_population
    from seednoise.data.datadecide import TRAITS
    from seednoise.experiments import (
        run_gain_calibration, run_leave_one_out, run_resplit,
    )

    pop, source = _population(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = set(args.part or ("gain", "resplit", "loo"))
    progress = None if args.quiet else True

    if "gain" in parts:
        g = run_gain_calibration(pop, grid=args.gain_grid, n_rep=args.n_rep,
                                 include_estimate=not args.gain_no_estimate,
                                 seed=args.seed, progress=progress)
        write_table(out, "gain_calibration", g["gain"])
        write_json(out, "gain_estimate", g["estimate"])
        est = g["estimate"]["gain_sd"]
        _log(f"estimated gain_sd = {est:.4f} (raw {g['estimate']['gain_sd_raw']:.4f})")
        for r in g["gain"]:
            if r["at_estimated_gain"] and r["phenotype"] == MARGIN:
                _log(f"N5 {r['spec']} at the estimated gain: Lambda={r['mean']:.4f}, "
                     f"share of excess={r['share_of_excess']:.3f}")

    if "resplit" in parts:
        if getattr(args, "synthetic", False):
            _log("resplit needs reduced runs on disk; skipped on a synthetic population")
        else:
            def rebuild(split_seed):
                return build_population(args.runs, TRAITS, n_runs=args.n_runs,
                                        split_seed=split_seed)[0]
            seeds = range(args.split_seed0, args.split_seed0 + args.n_splits)
            rows = run_resplit(rebuild, seeds, n_boot=args.n_boot, seed=args.seed,
                               progress=progress)
            write_table(out, "resplit", rows)
            for p in (MARGIN, ACCURACY):
                lam = [r["Lambda"] for r in rows if r["phenotype"] == p]
                neg = [r["negative_diagonals"] for r in rows if r["phenotype"] == p]
                _log(f"resplit {p}: Lambda {np.nanmin(lam):.4f} to {np.nanmax(lam):.4f} "
                     f"over {len(lam)} splits, negative diagonals in "
                     f"{sum(1 for n in neg if n > 0)} of them")

    if "loo" in parts:
        rows = run_leave_one_out(pop, recipes=source.get("recipes"),
                                 sizes=source.get("sizes"), n_boot=args.n_boot,
                                 seed=args.seed)
        write_table(out, "leave_one_out", rows)
        for kind in ("drop_recipe", "drop_size", "only_size"):
            lam = [r["Lambda"] for r in rows
                   if r["kind"] == kind and r["phenotype"] == MARGIN]
            _log(f"{kind} margin: Lambda {np.nanmin(lam):.4f} to {np.nanmax(lam):.4f}")
    write_json(out, "sensitivity_source", source)
    _log(f"tables written to {out}")
    return 0


def cmd_selftest(args):
    """End to end with no network: recover a known ``Lambda``, then run every stage.

    This is the check that the installed package works, so it asserts numbers
    rather than merely importing modules.
    """
    from seednoise.baselines import bakeoff
    from seednoise.estimator import estimate, k_eff
    from seednoise.gates import gate_table
    from seednoise.nulls import n1_calibration, n2_permutation, n5_gain_artifact
    from seednoise.simulate import default_spec, simulate

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fails, t0 = [], time.time()

    def check(name, ok, detail):
        _log(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
        if not ok:
            fails.append(name)

    n_items = None if args.full else tuple([600] * 10)
    rbar = 0.20
    spec = default_spec(rbar_e=rbar, n_config=400, seed=11)
    if n_items:
        spec.n_items = n_items
    pop = simulate(spec)
    target = float(np.sqrt(1.0 + (pop.K - 1) * rbar))

    e = estimate(pop, MARGIN, check=True)
    check("estimator recovers Lambda", abs(e.lambda_hat - target) < 0.12,
          f"{e.lambda_hat:.4f} against {target:.4f}, K_eff {k_eff(e.lambda_hat, pop.K):.2f}")

    spec0 = default_spec(rbar_e=0.0, n_config=400, seed=12)
    if n_items:
        spec0.n_items = n_items
    pop0 = simulate(spec0)
    e0 = estimate(pop0, MARGIN, check=True)
    check("null population sits at one", abs(e0.lambda_hat - 1.0) < 0.10,
          f"{e0.lambda_hat:.4f}")

    n1 = n1_calibration(rbars=(0.0, 0.2), n_rep=args.n_rep, n_config=85, seed=1,
                        which=("all",), n_items=n_items)
    for r in n1:
        if r["phenotype"] != MARGIN:
            continue
        tgt = float(np.sqrt(1.0 + 9 * r["rbar_E_true"]))
        check(f"N1 at rbar={r['rbar_E_true']}", abs(r["mean"] - tgt) < 0.06,
              f"mean {r['mean']:.4f} against {tgt:.4f}, sd {r['sd']:.4f}")

    n2 = n2_permutation(pop, MARGIN, n_rep=min(200, args.n_rep), seed=2)
    check("N2 permutation is centred at one", abs(n2["mean"] - 1.0) < 0.05,
          f"{n2['mean']:.4f} against an observed {n2['observed']:.4f}")

    n5 = n5_gain_artifact(gain_sd=0.05, n_rep=min(200, args.n_rep), n_config=85,
                          seed=3, n_items=n_items)
    lm = [r for r in n5 if r["phenotype"] == MARGIN][0]["mean"]
    la = [r for r in n5 if r["phenotype"] == ACCURACY][0]["mean"]
    check("N5 hits margin harder than accuracy", lm > la,
          f"margin {lm:.4f} against accuracy {la:.4f}")

    bo = bakeoff(pop, MARGIN, n_folds=3, seed=0)
    losses = {r["model"]: r["mse_log_sigma_agg"] for r in bo}
    check("bake-off ranks the structured models first",
          min(losses, key=losses.get) != "P0", f"winner {min(losses, key=losses.get)}")

    from seednoise.experiments import run_bakeoff, run_primary, run_screen
    small = pop.subset(np.arange(85))
    e1 = run_screen(small, n_estimation=small.N)
    e2 = run_primary(small, n_boot=199, seed=0)
    e4 = run_bakeoff(small, n_folds=3, seed=0)
    head = [r for r in e2["primary"] if r["label"] == "full contrast set"][0]
    check("intervals cover the truth",
          head["wild_lo"] <= target <= head["wild_hi"],
          f"[{head['wild_lo']:.3f}, {head['wild_hi']:.3f}] around {target:.3f}")

    write_table(out, "gates", gate_table([]) + e1["gates"])
    write_table(out, "primary", e2["primary"])
    write_table(out, "reliability", e1["reliability"])
    write_table(out, "bakeoff", e4["bakeoff"])
    write_json(out, "selftest", {"failures": fails, "lambda": e.lambda_hat,
                                 "target": target, "seconds": time.time() - t0})

    _log(f"{len(fails)} failures in {time.time() - t0:.1f}s; artifacts in {out}")
    return 1 if fails else 0


# --------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(
        prog="seednoise",
        description="Seed noise as the nonshared environment of a training run.")
    p.add_argument("--version", action="version", version=f"seednoise {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch recipe tarballs from the release")
    d.add_argument("--dest", default="tars")
    d.add_argument("--recipes", nargs="*", default=DEFAULT_RECIPES)
    d.add_argument("--quiet", action="store_true")
    d.set_defaults(func=cmd_download)

    r = sub.add_parser("reduce", help="reduce downloaded tarballs to per-run arrays")
    r.add_argument("--tars", default="tars")
    r.add_argument("--out", default="runs")
    r.add_argument("--sizes", nargs="*", default=None)
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_reduce)

    f = sub.add_parser("fetch", help="download and reduce one recipe at a time")
    f.add_argument("--tmp", default="tars")
    f.add_argument("--out", default="runs")
    f.add_argument("--recipes", nargs="*", default=DEFAULT_RECIPES)
    f.add_argument("--sizes", nargs="*", default=None)
    f.add_argument("--keep-tars", action="store_true")
    f.add_argument("--quiet", action="store_true")
    f.set_defaults(func=cmd_fetch)

    a = sub.add_parser("analyze", help="run E1 to E4 and write every table")
    a.add_argument("--runs", default="runs")
    a.add_argument("--out", default="results")
    a.add_argument("--n-runs", type=int, default=3)
    a.add_argument("--n-boot", type=int, default=4999)
    a.add_argument("--n-rep", type=int, default=2000)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--skip-nulls", action="store_true")
    a.add_argument("--synthetic", action="store_true",
                   help="analyse a simulated population instead of real runs")
    a.add_argument("--rbar", type=float, default=0.2)
    a.add_argument("--n-config", type=int, default=85)
    a.add_argument("--gain-sd", type=float, default=0.0)
    a.add_argument("--offset-sd", type=float, default=0.0)
    a.add_argument("--fast", action="store_true", help="fewer items per trait")
    a.add_argument("--arm2-runs", default=None,
                   help="reduced arm-2 runs, which add the G6 transport gate")
    a.add_argument("--arm2-n-runs", type=int, default=9)
    a.add_argument("--quiet", action="store_true")
    a.set_defaults(func=cmd_analyze)

    x = sub.add_parser("external",
                       help="the zero-cost check on the signal-and-noise release")
    x.add_argument("--parquet", default=None,
                   help="a downloaded random_seeds parquet; fetched if omitted")
    x.add_argument("--cache", default="cache/random_seeds.parquet")
    x.add_argument("--out", default="results")
    x.add_argument("--n-steps", type=int, default=20)
    x.set_defaults(func=cmd_external)

    m = sub.add_parser("arm2", help="score the PolyPythias seeds (needs a GPU)")
    m.add_argument("--out", default="runs-arm2")
    m.add_argument("--sizes", nargs="*", default=list(PP_SIZES))
    m.add_argument("--seeds", nargs="*", default=[str(s) for s in PP_SEEDS])
    m.add_argument("--n-per-task", type=int, default=500)
    m.add_argument("--revision", default=PP_REVISION)
    m.add_argument("--max-tokens", type=int, default=30_000)
    m.add_argument("--tasks", nargs="*", default=None)
    m.add_argument("--device", default=None)
    m.add_argument("--quiet", action="store_true")
    m.set_defaults(func=cmd_arm2)

    v = sub.add_parser("sensitivity",
                       help="E5: gain calibration, re-splits and leave-one-out")
    v.add_argument("--runs", default="runs")
    v.add_argument("--out", default="results")
    v.add_argument("--n-runs", type=int, default=3)
    v.add_argument("--part", nargs="*", choices=["gain", "resplit", "loo"],
                   default=None, help="which checks to run; all by default")
    v.add_argument("--gain-grid", nargs="*", type=float, default=list(GAIN_GRID))
    v.add_argument("--gain-no-estimate", action="store_true",
                   help="leave the data-driven gain out of the grid (for split runs)")
    v.add_argument("--n-rep", type=int, default=500)
    v.add_argument("--n-splits", type=int, default=50)
    v.add_argument("--split-seed0", type=int, default=1)
    v.add_argument("--n-boot", type=int, default=4999)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--synthetic", action="store_true")
    v.add_argument("--rbar", type=float, default=0.2)
    v.add_argument("--n-config", type=int, default=85)
    v.add_argument("--gain-sd", type=float, default=0.0)
    v.add_argument("--offset-sd", type=float, default=0.0)
    v.add_argument("--fast", action="store_true")
    v.add_argument("--quiet", action="store_true")
    v.set_defaults(func=cmd_sensitivity)

    s = sub.add_parser("selftest", help="offline end-to-end check of the install")
    s.add_argument("--out", default="selftest")
    s.add_argument("--n-rep", type=int, default=200)
    s.add_argument("--full", action="store_true",
                   help="use the real per-trait item counts (slower)")
    s.set_defaults(func=cmd_selftest)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        _log("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
