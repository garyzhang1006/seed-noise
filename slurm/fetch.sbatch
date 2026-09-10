#!/usr/bin/env bash
# Download and reduce the DataDecide release, one recipe per array task.
#
#   sbatch slurm/fetch.sbatch
#
# Each task streams one ~4.9 GB tarball into the node-local $TMPDIR, reduces it to
# per-run .npz arrays under $SEEDNOISE_ROOT/runs, and lets the tarball die with the
# job.  Peak scratch use is therefore the 48 MB of reduced runs, not the 122 GB
# release.  25 tasks at 25 concurrent is far under the 250-running-jobs cap.
# The tarballs come from huggingface.co over plain HTTPS (urllib, not the hub
# library), so HF_HUB_OFFLINE=1 from env.sh does not affect this job.  The
# compute nodes do have outbound network (the first run reached the hub from
# them); the login nodes cannot run the package at all, so there is no
# login-node fallback.
#SBATCH --job-name=sn-fetch
#SBATCH --partition=scu-cpu
#SBATCH --array=0-24%25
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --time=06:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/seed-noise/logs/%x-%A_%a.out
set -euo pipefail
ENV="${SEEDNOISE_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export SEEDNOISE_REPO" >&2; exit 2; }
. "$ENV"

RECIPE=$(python - "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys
from seednoise.data.datadecide import RECIPES
print(RECIPES[int(sys.argv[1])])
PY
)
PART="$SEEDNOISE_ROOT/runs-parts/$RECIPE"
# Recipe names are case-insensitive in the release, so match the file prefix loosely.
if [ "$(ls "$SEEDNOISE_ROOT/runs" 2>/dev/null | grep -ic "^${RECIPE}__")" -ge 15 ]; then
    echo "$RECIPE already has its 15 runs under $SEEDNOISE_ROOT/runs; nothing to do"; exit 0
fi
echo "task $SLURM_ARRAY_TASK_ID: recipe $RECIPE on $(hostname), tmp $TMPDIR"
# Each task reduces into its own directory so the per-invocation reduce_manifest
# is not overwritten by the other 24, then the arrays move to the shared runs/.
mkdir -p "$PART"
seednoise fetch --recipes "$RECIPE" --tmp "$TMPDIR/tars" --out "$PART" --quiet
mv -n "$PART"/*.npz "$SEEDNOISE_ROOT/runs/"
ls "$SEEDNOISE_ROOT/runs" | grep -c . | sed 's/^/reduced runs on disk: /'
