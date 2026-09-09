#!/usr/bin/env bash
# The E5 sensitivity checks, split the same way they were run on Kaggle: three
# tasks share the gain grid and one runs the re-splits and the leave-one-out sweep.
#
#   sbatch slurm/sensitivity.sbatch
#   sbatch --dependency=afterok:<fetch array job id> slurm/sensitivity.sbatch
#
# Then merge the gain tables:  python slurm/merge_sensitivity.py
# Each gain task is about 2000 simulated populations at 1.3 s each on a Kaggle core.
#SBATCH --job-name=sn-sens
#SBATCH --partition=scu-cpu
#SBATCH --array=0-3
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000M
#SBATCH --time=04:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/seed-noise/logs/%x-%A_%a.out
set -euo pipefail
ENV="${SEEDNOISE_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export SEEDNOISE_REPO" >&2; exit 2; }
. "$ENV"

OUT="$SEEDNOISE_ROOT/results-sens/part-$SLURM_ARRAY_TASK_ID"
COMMON=(--runs "$SEEDNOISE_ROOT/runs" --out "$OUT" --quiet)
case "$SLURM_ARRAY_TASK_ID" in
    0) seednoise sensitivity "${COMMON[@]}" --part gain --n-rep 500 --gain-grid 0.01 ;;
    1) seednoise sensitivity "${COMMON[@]}" --part gain --n-rep 500 --gain-grid 0.02 0.03 --gain-no-estimate ;;
    2) seednoise sensitivity "${COMMON[@]}" --part gain --n-rep 500 --gain-grid 0.05 0.10 --gain-no-estimate ;;
    3) seednoise sensitivity "${COMMON[@]}" --part resplit loo --n-splits 50 --n-boot 4999 ;;
esac
ls -la "$OUT"
