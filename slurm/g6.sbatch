#!/usr/bin/env bash
# The analysis again with arm 2 attached, which adds gate G6.
#
#   sbatch --dependency=afterok:<sn-arm2 array id>:<sn-analyze id> slurm/g6.sbatch
#
# Writes to a separate directory so the registered tables in results/ stay as
# they were; tab_gates.csv here is the one with the G6 row.
#SBATCH --job-name=sn-g6
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000M
#SBATCH --time=18:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/seed-noise/logs/%x-%j.out
set -euo pipefail
ENV="${SEEDNOISE_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export SEEDNOISE_REPO" >&2; exit 2; }
. "$ENV"

N=$(ls "$SEEDNOISE_ROOT/runs-arm2"/arm2__*.npz 2>/dev/null | wc -l)
[ "$N" -eq 27 ] || { echo "expected 27 arm-2 runs, found $N" >&2; exit 1; }
seednoise analyze --runs "$SEEDNOISE_ROOT/runs" --arm2-runs "$SEEDNOISE_ROOT/runs-arm2" \
    --out "$SEEDNOISE_ROOT/results-g6" --n-boot 4999 --n-rep 2000 --quiet
grep -E '^"?gate|G6' "$SEEDNOISE_ROOT/results-g6/tab_gates.csv"
