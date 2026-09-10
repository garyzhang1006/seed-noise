#!/usr/bin/env bash
# Merge the four sensitivity parts into results/, after the sn-sens array.
#
#   sbatch --dependency=afterok:<sn-sens array id> slurm/merge.sbatch
#
# A job rather than a login-node command so that it runs with the venv's python;
# the script itself is plain enough to run under the login nodes' 3.6 as well.
#SBATCH --job-name=sn-merge
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=00:10:00
#SBATCH --output=/athena/accardilab/scratch/%u/seed-noise/logs/%x-%j.out
set -euo pipefail
REPO="${SEEDNOISE_REPO:-$SLURM_SUBMIT_DIR}"
ENV="$REPO/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export SEEDNOISE_REPO" >&2; exit 2; }
. "$ENV"
python "$REPO/slurm/merge_sensitivity.py" --root "$SEEDNOISE_ROOT"
