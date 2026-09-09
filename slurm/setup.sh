#!/usr/bin/env bash
# One-off install on a login node.  Creates the virtualenv on scratch, installs
# this checkout with the gpu, hub and dev extras, and runs the offline self-test.
#
#   bash slurm/setup.sh
#
# The torch wheel carries its own CUDA runtime, so no cuda module is needed
# either here or in the jobs; the cluster's cuda/13.0 module only supplies nvcc.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export SEEDNOISE_ROOT="${SEEDNOISE_ROOT:-/athena/accardilab/scratch/$USER/seed-noise}"
export SEEDNOISE_VENV="${SEEDNOISE_VENV:-$SEEDNOISE_ROOT/venv}"
mkdir -p "$SEEDNOISE_ROOT"/{runs,runs-arm2,results,results-sens,logs}

if [ ! -f "$SEEDNOISE_VENV/bin/activate" ]; then
    python3 -c 'import sys; assert sys.version_info >= (3, 9), sys.version' \
        || { echo "python3 on this node is older than 3.9; load a newer python module first" >&2; exit 1; }
    python3 -m venv "$SEEDNOISE_VENV"
fi
# shellcheck disable=SC1090
. "$SEEDNOISE_VENV/bin/activate"
pip install -q --upgrade pip
pip install -q -e "$REPO[gpu,hub,dev]"
seednoise --version
seednoise selftest --out "$SEEDNOISE_ROOT/selftest"
echo "installed into $SEEDNOISE_VENV; jobs write under $SEEDNOISE_ROOT"
