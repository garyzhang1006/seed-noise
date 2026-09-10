#!/usr/bin/env bash
# One-off install.  Creates the virtualenv on scratch, installs this checkout
# with the gpu, hub and dev extras, and runs the offline self-test.
#
#   bash slurm/setup.sh
#
# The login nodes run CentOS 7 with a system python3 of 3.6.8, which cannot even
# compile this package, while the compute nodes' /usr/bin/python3 is 3.9.  So when
# the python found here is too old the script hands itself to an interactive
# scu-cpu job with srun and builds the venv there.  A venv built on a compute node
# works on the login nodes only for `sbatch` and reading logs, which is all the
# login node needs to do; every python step below runs inside a job.
#
# The torch wheel carries its own CUDA runtime, so no cuda module is needed
# either here or in the jobs; the cluster's cuda/13.0 module only supplies nvcc.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export SEEDNOISE_ROOT="${SEEDNOISE_ROOT:-/athena/accardilab/scratch/$USER/seed-noise}"
export SEEDNOISE_VENV="${SEEDNOISE_VENV:-$SEEDNOISE_ROOT/venv}"
mkdir -p "$SEEDNOISE_ROOT"/{runs,runs-arm2,results,results-sens,logs}

py_ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; }

if [ -z "${SLURM_JOB_ID:-}" ] && ! py_ok python3; then
    if ! command -v srun >/dev/null; then
        echo "python3 here is $(python3 --version 2>&1) and there is no srun; install with a python >= 3.9" >&2
        exit 1
    fi
    echo "python3 on this node is $(python3 --version 2>&1); building the venv inside an scu-cpu job"
    exec srun --partition=scu-cpu --cpus-per-task=2 --mem=8000M --time=01:00:00 \
        --job-name=sn-setup --export=ALL bash "$REPO/slurm/setup.sh"
fi
py_ok python3 || { echo "python3 is $(python3 --version 2>&1) even on the compute node; load a python >= 3.9 module" >&2; exit 1; }

if [ ! -f "$SEEDNOISE_VENV/bin/activate" ]; then
    python3 -m venv "$SEEDNOISE_VENV"
fi
# shellcheck disable=SC1090
. "$SEEDNOISE_VENV/bin/activate"
py_ok python || { echo "the venv at $SEEDNOISE_VENV was built with $(python --version 2>&1); delete it and rerun" >&2; exit 1; }
pip install -q --upgrade pip
pip install -q -e "$REPO[gpu,hub,dev]"
seednoise --version
seednoise selftest --out "$SEEDNOISE_ROOT/selftest"
echo "installed into $SEEDNOISE_VENV; jobs write under $SEEDNOISE_ROOT"
