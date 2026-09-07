# Running the pipeline on Kaggle CPU

The two scripts here are the kernels that produced the DataDecide tables in the
paper. Kaggle cannot clone a private repository, so the package travels as a
dataset instead.

1. Upload the source as a dataset once, and re-version it after every change:
   `tar czf seed-noise-src.tar.gz --exclude=.git seed-noise` next to a
   `dataset-metadata.json` naming `<user>/seed-noise-src`, then
   `kaggle datasets create -p .` or `kaggle datasets version -p . -m msg`.
   Kaggle unpacks the tarball and mounts it under
   `/kaggle/input/datasets/<user>/seed-noise-src/`, which the scripts locate by
   searching for `pyproject.toml`.
2. Push `fetch.py` as a script kernel with `dataset_sources` set to that dataset
   and `enable_internet` on. Each recipe takes about 200 seconds to reduce, so the
   25 recipes split across three kernels finish in about 36 minutes. Edit
   `RECIPES` per kernel.
3. Push `analyze.py` with `kernel_sources` listing the fetch kernels. It gathers
   every `.npz` under `/kaggle/input`, runs `seednoise analyze` at the registered
   settings and `seednoise external`, and writes `results/`. The registered run
   with `--n-rep 2000` took about six hours on a Kaggle core; `--skip-nulls`
   finishes within the hour and leaves the null-based gates out of `tab_gates.csv`.

Retrieve tables with the Python API rather than the CLI, which drops the
connection on the 375 reduced runs:

```python
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
api.kernels_output("<user>/seed-noise-analyze", "out", file_pattern=r"(\.log$|results/)",
                   force=True, quiet=True, page_size=500)
```
