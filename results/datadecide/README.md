# DataDecide tables

Produced on Kaggle CPU from the 375 DataDecide runs (25 recipes, 5 sizes, 3
seeds) by `seednoise analyze` at the registered settings, `--n-boot 4999` and
`--n-rep 2000`, with `split_seed` 20260101, followed by `seednoise external`.
The registered run took 21503 seconds on one Kaggle core, of which the nulls
account for all but about twenty seconds. A `--skip-nulls` rerun reproduced
every shared table byte for byte, and an earlier run that crashed in the
bake-off before commit 3d7ad3c reproduced the primary and null tables byte for
byte as well.

`tab_gates.csv` holds the seven gates the pipeline can evaluate on this data.
G3 fails because winogrande has a negative cross-half diagonal, G7 fails because
DataDecide carries no held-out-loss field, and G5 fails because the N5 gain
artefact at gain_sd 0.05 manufactures 0.148 of the observed 0.244 excess on the
margin, a share of 0.61 against the registered ceiling of 0.5. G6 needs the
second arm and is not evaluated here.
