# DataDecide tables

Produced on Kaggle CPU from the 375 DataDecide runs (25 recipes, 5 sizes, 3
seeds) by `seednoise analyze` at the registered settings, `--n-boot 4999` and
`--n-rep 2000`, with `split_seed` 20260101, followed by `seednoise external`.
The registered run took 21503 seconds on one Kaggle core, of which the nulls
account for all but about twenty seconds. A `--skip-nulls` rerun reproduced
every shared table byte for byte, and an earlier run that crashed in the
bake-off before commit 3d7ad3c reproduced the primary and null tables byte for
byte as well. The 375 reduced runs (48 MB of `.npz`, one per run) are kept
outside the repository as the private Kaggle dataset
`garyzhang11111/seed-noise-reduced-runs`, alongside the source dataset
`garyzhang11111/seed-noise-src` the kernels install from.

`tab_gates.csv` holds the seven gates the pipeline can evaluate on this data.
G3 fails because winogrande has a negative cross-half diagonal, G7 fails because
DataDecide carries no held-out-loss field, and G5 fails because the N5 gain
artefact at gain_sd 0.05 manufactures 0.148 of the observed 0.244 excess on the
margin, a share of 0.61 against the registered ceiling of 0.5. G6 needs the
second arm and is not evaluated here.

## Sensitivity (E5)

`seednoise sensitivity` adds three checks that the registered run does not
cover, all from the same 375 runs and the same reduced arrays. The gain
calibration ran as three Kaggle kernels that split the grid (`--gain-grid 0.01`
with the data-driven point, `0.02 0.03` and `0.05 0.10`, the latter two with
`--gain-no-estimate`), 500 replicates per point and spec, about 1800 seconds
each, and `tab_gain_calibration.csv` is their concatenation. The re-splits and
the leave-one-out sweep ran in one kernel in 256 seconds.

`gain_estimate.json` gives the across-seed spread of a run-level multiplicative
gain estimated from the data, 0.042 after subtracting the leakage of
independent seed effects into the fit (0.063 before). Seed effects shared
across traits in proportion to the mean profile cannot be told from a gain at
the population level, so the number is an upper bound. At that gain null N5
manufactures a Lambda of 1.108 on the registered simulator, a share of 0.443 of
the observed excess, and 1.014 on a simulator matched to the data's margin
levels, seed variances and item noise (`matched_spec`), a share of 0.059. The
G5 failure in `tab_gates.csv` therefore comes from the registered gain of 0.05
sitting above the data's own spread on a simulator whose margins sit about four
seed standard deviations from zero, and the check passes at the estimated gain
on both specs. The accuracy rows are identical across the grid because a
multiplicative gain cannot move a margin across zero, which is a consistency
check on the simulator rather than a finding.

`tab_resplit.csv` rebuilds the population under 50 fresh item splits
(`split_seed` 1 to 50). The margin Lambda ranges from 1.243 to 1.252 with the
wild lower endpoint above 1 in all 50, accuracy from 1.066 to 1.107, and the
winogrande diagonal is negative in 49 of the 50 splits for both phenotypes, so
the G3 failure is a property of the trait and not of the registered split.

`tab_leave_one_out.csv` drops each recipe (margin 1.207 to 1.256, wild lower
endpoint above 1 in all 25), drops each size band (1.178 to 1.289, lower
endpoint above 1 in 4 of 5) and keeps each size band alone (25 configurations,
1.101 to 1.514). The single-size estimates are too wide to rank sizes, and the
1B margin row has no lower endpoint because the bootstrap-t lower bound of the
squared ratio fell below zero.
