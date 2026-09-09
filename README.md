# seed-noise

Code for *The seed is the nonshared environment*, which treats the training seed
of a language model the way behaviour genetics treats the part of the environment
siblings do not share, and estimates the nonshared-environmental correlation
matrix `R_E` across a battery of benchmarks without any measurement-error
correction to argue about.

The package is called `seednoise`. It reduces the DataDecide per-instance release
to two phenotypes per item, splits every benchmark into two disjoint halves,
estimates `Sigma_E` from the cross-half product of within-configuration
deviations, and runs the four experiments the paper reports. Every number in the
paper comes out of `seednoise` and nothing is hand-transcribed.

## What the code computes

Each run is scored on 37,682 items across ten traits. For item `i` with gold
answer `g`, the two co-primary phenotypes are the per-byte margin
`m = l(g)/bytes(g) - max_{k != g} l(k)/bytes(k)` and the accuracy `1[m > 0]`, so
accuracy is the thresholded image of the margin and both come from one pass over
the same records. Trait scores are item means, with MMLU macro-averaged over its
57 subjects rather than pooled over items.

The estimator rests on one identity. Split the items of every trait into halves A
and B once, score both halves of every run, and the expected product of the
within-configuration deviations across halves is `(1 - 1/R) Sigma_E(j,k)`, with
the item noise gone because the two halves are disjoint samples. Dividing by
`R - 1` returns `Sigma_E` exactly, including its diagonal, which is what makes the
whole analysis free of a reliability correction.

The headline is a single ratio,

    Lambda_hat = sqrt( sum_c T_c / sum_c U_c ),

with `T_c` the cross-half product of the aggregate contrast and `U_c` the mean of
the same product taken trait by trait. `Lambda^2 = 1 + (K-1) rbar_E` and
`K_eff = K / Lambda^2`, so `Lambda = 1` is exactly the null of independent
per-trait seed effects and needs no calibration constant. Everything is an average
over an orthonormal basis of the mean-zero contrast space, and because every basis
row is orthogonal to the all-ones vector, the raw-score failure that Remark 2
describes cannot occur here.

Two confounds get their own machinery. The token budget is handled by scoring
every configuration at the largest step all of its seeds reached, and by a
batch-free contrast that lives entirely inside the auxiliary seeds. Run-level
sharpness is handled by mediation on a leave-one-trait-out competence proxy and on
the run's mean per-byte log-likelihood, one pooled slope per trait.

Inference treats the configuration as the unit and the recipe as the cluster: a
wild cluster bootstrap-t over 17 recipes with `t(16)` critical values, a
configuration bootstrap beside it, and `DEFF = 1 + 4 rho_ICC` reported so the
effective sample size is visible.

## Install

```bash
pip install -e ".[dev]"
```

The base install needs only numpy, scipy, pandas, pyyaml and tqdm, which is enough
for every estimation stage on runs somebody else reduced. The `gpu` extra pulls
torch, transformers and datasets for arm 2; the `hub` extra pulls pyarrow for the
external check.

## Check the install before spending anything

```bash
seednoise selftest
```

This simulates a population whose `Lambda` is known, recovers it, runs the nulls,
the bake-off and the intervals, and exits non-zero if any of the eight checks
fail. It needs no data, no GPU and no network, and it finishes in about twenty
seconds. A recent run:

```
ok   estimator recovers Lambda: 1.7294 against 1.6733, K_eff 3.34
ok   null population sits at one: 1.0165
ok   N1 at rbar=0.0: mean 0.9991 against 1.0000, sd 0.0625
ok   N1 at rbar=0.2: mean 1.6751 against 1.6733, sd 0.0738
ok   N2 permutation is centred at one: 0.9984 against an observed 1.7294
ok   N5 hits margin harder than accuracy: margin 1.1523 against accuracy 1.0018
ok   bake-off ranks the structured models first: winner P3
ok   intervals cover the truth: [1.549, 1.808] around 1.673
```

The test suite is the stronger check and takes about half a minute:

```bash
pytest -q
```

Every test asserts an identity or an invariant rather than an implementation
detail: that the cross-half estimator is unbiased where an ordinary
within-configuration covariance is not, that the batch-free contrast is blind to a
planted batch offset, that the wild bootstrap covers at its nominal rate over 17
clusters, and that the information ratio at `p = 0.35` reproduces the 1.66 the
paper claims.

## The main run, end to end

The release is 25 recipe tarballs of about 4.9 GB each, so `fetch` downloads one,
reduces it and deletes it before starting the next. Peak disk is one tarball plus
the reduced runs, which come to roughly 30 MB for the whole population.

```bash
seednoise fetch --out runs            # ~122 GB of sequential download
seednoise analyze --runs runs --out results
```

`analyze` writes one CSV per table the paper cites, plus the matrices as JSON:
`tab_primary`, `tab_practitioner`, `tab_mediation`, `tab_reliability`,
`tab_nulls`, `tab_bakeoff`, `tab_cheverud`, `tab_gates`, `matrices_margin.json`,
`matrices_accuracy.json` and `source.json`. It also prints which gates failed,
which is the only thing worth reading first.

To rehearse the whole analysis before any download, run it on a synthetic
population whose truth you set:

```bash
seednoise analyze --synthetic --fast --rbar 0.15 --out results-dry
```

That takes about a minute and exercises every stage, including the nulls. G7 fails
by construction there, because a simulated population has no held-out-loss field.

## The two side arms

The external check costs nothing and needs no GPU. Heineman et al.'s
signal-and-noise release includes ten OLMo2 1B runs that differ only in the
training seed, so the same estimand can be read off an independent lab's runs:

```bash
seednoise external --out results
```

The released scores are aggregates, so there is no half split and the correlations
are attenuated by item noise; every `Lambda` in that table is therefore a floor
rather than an estimate. On the real release the training-seed runs give
`rbar = 0.112` and `Lambda = 1.70` on bits-per-byte over 18 tasks, and 1.46 after
the leave-one-task-out competence control, while the coarser accuracy metric lands
at 1.

Arm 2 is the transport check and is the only stage that needs an accelerator. It
scores the nine PolyPythias replicate seeds at 70M, 160M and 410M on a nested
500-items-per-task subsample of the same battery:

```bash
seednoise arm2 --out runs-arm2 --n-per-task 500
seednoise analyze --runs runs --arm2-runs runs-arm2 --out results
```

Passing `--arm2-runs` adds gate G6, which asks whether the excess `Lambda - 1`
transports to within a fifth. The subsample is nested rather than merely random, so
a 200-item pilot is a strict subset of the 500-item run and the two are comparable.

## Notebooks

`notebooks/` holds three Kaggle notebooks: one that fetches and reduces the
release on CPU, one that runs the whole analysis on the reduced runs, and one that
scores arm 2 on T4 x2. They install the package from this repository and write
their outputs to `/kaggle/working`.

## Slurm

`slurm/` holds sbatch scripts for the SCU cluster (Slurm, `scu-cpu` and
`scu-gpu` partitions, Lustre scratch under `/athena/accardilab/scratch`), with
the same split as the notebooks: a CPU array that fetches and reduces the
release, the registered analysis, the E5 sensitivity checks, a 27-task GPU array
for arm 2 and the G6 analysis on top of it. `bash slurm/pipeline.sh` submits the
lot with dependencies; `slurm/README.md` explains the resource choices.

## Layout

    src/seednoise/
      phenotypes.py    per-choice scores to per-item margin and accuracy
      halves.py        the once-and-for-all A/B item split
      population.py    the (N, R, K) arrays every estimator reads
      estimator.py     contrast basis, Sigma_E, Lambda, K_eff, P_flip
      reliability.py   variance components, information ratio, power
      inference.py     cluster-t, wild cluster bootstrap-t, config bootstrap
      mediation.py     the competence and sharpness mediators
      baselines.py     P0 to P3 and the out-of-sample bake-off
      nulls.py         N1 to N6
      gates.py         G0 to G8, each with the fallback it triggers
      simulate.py      the synthetic population the nulls are calibrated on
      build.py         reduced runs to a Population
      store.py         float16 margins and bit-packed accuracy on disk
      data/            DataDecide, PolyPythias and signal-and-noise readers
      experiments/     E1 to E4, which produce the tables

## Data

DataDecide's per-instance outputs are at
`allenai/DataDecide-eval-instances`; the PolyPythias seeds are the
`EleutherAI/pythia-<size>-seed<k>` models; the external check reads
`allenai/signal-and-noise`. Nothing in this repository redistributes any of them.
