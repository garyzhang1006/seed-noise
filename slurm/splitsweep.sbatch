#!/usr/bin/env bash
# Every 17-of-25 estimation-recipe subset, from the same reduced runs.
#
#   sbatch --dependency=afterok:<fetch array job id> slurm/splitsweep.sbatch
#
# The registration described eight screening recipes and 17 estimation recipes,
# a partition the code never implemented; every registered table uses all 25.
# This job enumerates the 1081575 subsets of 17 recipes so the paper can report
# the range the headline would have taken under any of them.  Per-configuration
# T and U do not depend on which other configurations are present, so the sweep
# is one population build plus arithmetic: minutes on one core.
#SBATCH --job-name=sn-splitsweep
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000M
#SBATCH --time=01:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/seed-noise/logs/%x-%j.out
set -euo pipefail
ENV="${SEEDNOISE_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export SEEDNOISE_REPO" >&2; exit 2; }
. "$ENV"

seednoise splitsweep --runs "$SEEDNOISE_ROOT/runs" --out "$SEEDNOISE_ROOT/results" --quiet
cat "$SEEDNOISE_ROOT/results/tab_splitsweep.csv"
