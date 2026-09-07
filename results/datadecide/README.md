# DataDecide tables

Produced on Kaggle CPU from the 375 DataDecide runs (25 recipes, 5 sizes, 3
seeds) at the registered settings, `--n-boot 4999` and `--n-rep 2000`, with
`split_seed` 20260101. `tab_nulls.csv` comes from the first analyze kernel, which
finished the nulls and then crashed in the bake-off on a fold with an undefined
correlation (fixed in commit 3d7ad3c). The other tables come from the rerun with
`--skip-nulls` after that fix; the estimator, the bootstrap and the bake-off are
seeded and do not depend on the nulls, so the two runs agree on every shared
table. `tab_gates.csv` therefore lists the data gates only; the null-based
gates will be appended when the full registered rerun completes.
