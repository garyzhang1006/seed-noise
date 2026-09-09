#!/usr/bin/env bash
# The registered analysis on the reduced runs, then the external check.
#
#   sbatch slurm/analyze.sbatch
#   sbatch --dependency=afterok:<fetch array job id> slurm/analyze.sbatch
#
# One core: the nulls dominate and the code is single-threaded, so more cores buy
# nothing.  The registered settings took 21503 s on a Kaggle core; the time limit
# leaves room for a slower node.
#SBATCH --job-name=sn-analyze
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000M
#SBATCH --time=18:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/seed-noise/logs/%x-%j.out
set -euo pipefail
ENV="${SEEDNOISE_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export SEEDNOISE_REPO" >&2; exit 2; }
. "$ENV"

N_RUNS=$(ls "$SEEDNOISE_ROOT/runs"/*.npz 2>/dev/null | wc -l)
[ "$N_RUNS" -eq 375 ] || echo "warning: expected 375 reduced runs, found $N_RUNS" >&2
seednoise analyze --runs "$SEEDNOISE_ROOT/runs" --out "$SEEDNOISE_ROOT/results" \
    --n-boot 4999 --n-rep 2000 --quiet
seednoise external --out "$SEEDNOISE_ROOT/results" || echo "external check failed (rc $?)" >&2
cat "$SEEDNOISE_ROOT/results/tab_gates.csv"
